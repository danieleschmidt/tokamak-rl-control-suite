# Tokamak RL Control Suite - Infrastructure as Code

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "tokamak-rl-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "tokamak-rl-control-suite"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "tokamak-rl-${var.environment}"
  cluster_version = "1.27"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    main = {
      name = "tokamak-rl-nodes"

      instance_types = ["m5.xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 2
      max_size     = 10
      desired_size = 3

      labels = {
        role = "tokamak-rl"
      }

      tags = {
        ExtraTag = "tokamak-rl-nodes"
      }
    }
    
    gpu = {
      name = "tokamak-rl-gpu-nodes"

      instance_types = ["p3.2xlarge"]
      capacity_type  = "SPOT"

      min_size     = 0
      max_size     = 5
      desired_size = 1

      labels = {
        role = "tokamak-rl-gpu"
      }

      taints = {
        nvidia-gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "tokamak-rl-vpc-${var.environment}"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# RDS Database
resource "aws_db_instance" "tokamak_rl" {
  identifier = "tokamak-rl-${var.environment}"

  allocated_storage    = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  db_name  = "tokamak_rl"
  username = "tokamak_rl_user"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.tokamak_rl.name

  backup_retention_period = 7
  backup_window          = "07:00-09:00"
  maintenance_window     = "Sun:09:00-Sun:11:00"

  deletion_protection = true
  skip_final_snapshot = false
  
  performance_insights_enabled = true
  monitoring_interval         = 60

  tags = {
    Name = "tokamak-rl-database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "tokamak_rl" {
  name       = "tokamak-rl-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "tokamak_rl" {
  description       = "Redis cluster for Tokamak RL"
  replication_group_id = "tokamak-rl-${var.environment}"

  port               = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  node_type          = "cache.t3.micro"
  
  subnet_group_name  = aws_elasticache_subnet_group.tokamak_rl.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  automatic_failover_enabled = true
  multi_az_enabled          = true

  snapshot_retention_limit = 3
  snapshot_window         = "07:00-09:00"

  tags = {
    Name = "tokamak-rl-redis"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "tokamak_rl_data" {
  bucket = "tokamak-rl-data-${var.environment}-${random_string.suffix.result}"

  tags = {
    Name = "tokamak-rl-data"
  }
}

resource "aws_s3_bucket" "tokamak_rl_models" {
  bucket = "tokamak-rl-models-${var.environment}-${random_string.suffix.result}"

  tags = {
    Name = "tokamak-rl-models"
  }
}

resource "random_string" "suffix" {
  length  = 8
  upper   = false
  special = false
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "tokamak-rl-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "tokamak-rl-rds-sg"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "tokamak-rl-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "tokamak-rl-redis-sg"
  }
}

resource "aws_db_subnet_group" "tokamak_rl" {
  name       = "tokamak-rl-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "tokamak-rl-db-subnet-group"
  }
}
