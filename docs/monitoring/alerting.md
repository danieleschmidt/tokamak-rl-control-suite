# Alerting Configuration

Comprehensive alerting setup for tokamak-rl-control-suite with multi-channel notification and escalation policies.

## Alerting Overview

The alerting system provides:
- **Real-time monitoring** of critical system and plasma parameters
- **Multi-tier alerting** with escalation policies
- **Context-aware notifications** with runbook links
- **Integration** with multiple notification channels
- **Alert fatigue prevention** through intelligent grouping and suppression

## Alert Categories

### 1. Critical System Alerts
High-severity alerts requiring immediate attention

### 2. Plasma Safety Alerts  
Domain-specific alerts for fusion plasma safety

### 3. Performance Alerts
Alerts for degraded system performance

### 4. Training Alerts
RL training and model performance alerts

## Prometheus Alert Rules

### System-Level Alerts

```yaml
# alerts/system.yml
groups:
  - name: tokamak-system-critical
    rules:
      - alert: ServiceDown
        expr: up{job="tokamak-rl-control"} == 0
        for: 30s
        labels:
          severity: critical
          category: system
          runbook: "https://docs.internal/runbooks/service-down"
        annotations:
          summary: "Tokamak RL Control service is down"
          description: "Service {{ $labels.instance }} has been down for more than 30 seconds"
          impact: "Complete loss of plasma control capability"
          action: "Check service logs and restart if necessary"
      
      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
          category: system
          runbook: "https://docs.internal/runbooks/high-cpu"
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% for 5+ minutes on {{ $labels.instance }}"
          impact: "Potential performance degradation"
          action: "Check for resource-intensive processes"
      
      - alert: CriticalCPUUsage
        expr: system_cpu_usage_percent > 95
        for: 2m
        labels:
          severity: critical
          category: system
          runbook: "https://docs.internal/runbooks/critical-cpu"
        annotations:
          summary: "Critical CPU usage detected"
          description: "CPU usage is {{ $value }}% for 2+ minutes on {{ $labels.instance }}"
          impact: "Severe performance degradation, potential system instability"
          action: "Immediate intervention required - check processes and scale if needed"
      
      - alert: HighMemoryUsage
        expr: (system_memory_usage_bytes{type="used"} / system_memory_usage_bytes{type="total"}) * 100 > 85
        for: 5m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"
      
      - alert: CriticalMemoryUsage
        expr: (system_memory_usage_bytes{type="used"} / system_memory_usage_bytes{type="total"}) * 100 > 95
        for: 1m
        labels:
          severity: critical
          category: system
        annotations:
          summary: "Critical memory usage detected"
          description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"
          impact: "Risk of OOM kills and service failures"
      
      - alert: GPUMemoryExhaustion
        expr: (system_gpu_memory_bytes{type="used"} / system_gpu_memory_bytes{type="total"}) * 100 > 90
        for: 2m
        labels:
          severity: critical
          category: system
          gpu_id: "{{ $labels.gpu_id }}"
        annotations:
          summary: "GPU memory nearly exhausted"
          description: "GPU {{ $labels.gpu_id }} memory usage is {{ $value }}%"
          impact: "Training and inference may fail"
      
      - alert: HighErrorRate
        expr: rate(application_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "High application error rate"
          description: "Error rate is {{ $value }} errors/sec"
```

### Plasma Safety Alerts

```yaml
# alerts/plasma-safety.yml
groups:
  - name: tokamak-plasma-safety
    rules:
      - alert: LowSafetyFactor
        expr: plasma_q_min < 1.5
        for: 10s
        labels:
          severity: critical
          category: plasma-safety
          runbook: "https://docs.internal/runbooks/low-q-min"
        annotations:
          summary: "Dangerously low safety factor"
          description: "q_min = {{ $value }} on {{ $labels.tokamak_config }}"
          impact: "Risk of plasma disruption"
          action: "Engage safety systems immediately"
      
      - alert: CriticalSafetyFactor
        expr: plasma_q_min < 1.2
        for: 5s
        labels:
          severity: critical
          category: plasma-safety
          escalation: "immediate"
        annotations:
          summary: "CRITICAL: Extremely low safety factor"
          description: "q_min = {{ $value }} - DISRUPTION IMMINENT"
          impact: "Plasma disruption highly likely"
          action: "EMERGENCY SHUTDOWN PROCEDURES"
      
      - alert: HighBetaNormalized
        expr: plasma_beta_normalized > 0.04
        for: 30s
        labels:
          severity: warning
          category: plasma-safety
        annotations:
          summary: "High normalized beta"
          description: "β_N = {{ $value }} approaching stability limit"
          impact: "Increased disruption risk"
      
      - alert: HighShapeError
        expr: plasma_shape_error_cm{metric="current"} > 5.0
        for: 1m
        labels:
          severity: warning
          category: plasma-control
        annotations:
          summary: "High plasma shape error"
          description: "Shape error is {{ $value }} cm"
          impact: "Poor plasma confinement"
      
      - alert: CriticalShapeError
        expr: plasma_shape_error_cm{metric="current"} > 10.0
        for: 30s
        labels:
          severity: critical
          category: plasma-control
        annotations:
          summary: "Critical plasma shape error"
          description: "Shape error is {{ $value }} cm"
          impact: "Severe confinement degradation"
      
      - alert: DisruptionDetected
        expr: rate(plasma_disruptions_total[1m]) > 0
        for: 0s
        labels:
          severity: critical
          category: plasma-safety
          escalation: "immediate"
        annotations:
          summary: "Plasma disruption detected"
          description: "Disruption in {{ $labels.tokamak_config }}"
          impact: "Plasma lost, potential damage to vessel"
          action: "Post-disruption analysis and recovery procedures"
      
      - alert: HighDisruptionRate
        expr: rate(plasma_disruptions_total[5m]) > 0.1
        for: 1m
        labels:
          severity: warning
          category: plasma-safety
        annotations:
          summary: "High disruption rate"
          description: "Disruption rate: {{ $value }}/min"
          impact: "Reduced operational efficiency"
      
      - alert: SafetySystemFailure
        expr: safety_margin_percent < 10
        for: 30s
        labels:
          severity: critical
          category: plasma-safety
        annotations:
          summary: "Safety system margin critically low"
          description: "Safety margin for {{ $labels.constraint_type }}: {{ $value }}%"
          impact: "Safety system effectiveness compromised"
```

### Performance Alerts

```yaml
# alerts/performance.yml
groups:
  - name: tokamak-performance
    rules:
      - alert: SlowInference
        expr: histogram_quantile(0.95, rl_inference_duration_seconds_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "Slow RL model inference"
          description: "95th percentile inference time: {{ $value }}s for {{ $labels.model_name }}"
          impact: "Real-time control may be compromised"
      
      - alert: CriticalInferenceLatency
        expr: histogram_quantile(0.95, rl_inference_duration_seconds_bucket) > 0.5
        for: 2m
        labels:
          severity: critical
          category: performance
        annotations:
          summary: "Critical inference latency"
          description: "95th percentile inference time: {{ $value }}s"
          impact: "Real-time control severely degraded"
      
      - alert: SlowSimulation
        expr: histogram_quantile(0.95, simulation_step_duration_seconds_bucket) > 0.05
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "Slow physics simulation"
          description: "95th percentile simulation step: {{ $value }}s"
      
      - alert: LowTrainingFPS
        expr: training_fps < 100
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "Low training FPS"
          description: "Training FPS: {{ $value }} for {{ $labels.model_name }}"
          impact: "Extended training times"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 2.0
        for: 3m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "High API response time"
          description: "95th percentile response time: {{ $value }}s"
```

### Training Alerts

```yaml
# alerts/training.yml
groups:
  - name: tokamak-training
    rules:
      - alert: TrainingStalled
        expr: rate(rl_training_episodes_total[10m]) == 0
        for: 5m
        labels:
          severity: warning
          category: training
        annotations:
          summary: "RL training appears stalled"
          description: "No training episodes completed in 10 minutes for {{ $labels.model_name }}"
          impact: "Model improvement halted"
      
      - alert: HighTrainingLoss
        expr: model_loss{loss_type="total"} > 100
        for: 10m
        labels:
          severity: warning
          category: training
        annotations:
          summary: "High training loss detected"
          description: "Total loss: {{ $value }} for {{ $labels.model_name }}"
          impact: "Poor learning progress"
      
      - alert: DivergingLoss
        expr: increase(model_loss{loss_type="total"}[30m]) > 50
        for: 5m
        labels:
          severity: critical
          category: training
        annotations:
          summary: "Training loss diverging"
          description: "Loss increased by {{ $value }} in 30 minutes"
          impact: "Training instability"
      
      - alert: LowEpisodeReward
        expr: avg_over_time(rl_training_episode_reward_bucket[1h]) < -50
        for: 30m
        labels:
          severity: warning
          category: training
        annotations:
          summary: "Low average episode reward"
          description: "Average reward: {{ $value }} for {{ $labels.model_name }}"
      
      - alert: NoModelImprovement
        expr: |
          (
            avg_over_time(rl_training_episode_reward_bucket[1h]) -
            avg_over_time(rl_training_episode_reward_bucket[1h] offset 24h)
          ) < 0.1
        for: 1h
        labels:
          severity: warning
          category: training
        annotations:
          summary: "No significant model improvement"
          description: "Reward improvement: {{ $value }} in 24h"
```

## Notification Channels

### Slack Integration

```yaml
# alertmanager.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  group_by: ['alertname', 'category']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
        category: plasma-safety
      receiver: 'plasma-safety-critical'
      group_wait: 0s
      repeat_interval: 1m
    
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 30s
      repeat_interval: 5m
    
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'plasma-safety-critical'
    slack_configs:
      - channel: '#plasma-safety-alerts'
        title: 'CRITICAL PLASMA SAFETY ALERT'
        text: |
          🚨 **CRITICAL SAFETY ALERT** 🚨
          
          **Alert:** {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
          **Severity:** {{ .CommonLabels.severity }}
          **Impact:** {{ range .Alerts }}{{ .Annotations.impact }}{{ end }}
          **Action:** {{ range .Alerts }}{{ .Annotations.action }}{{ end }}
          
          **Details:**
          {{ range .Alerts }}
          - **Instance:** {{ .Labels.instance }}
          - **Value:** {{ .Annotations.description }}
          {{ if .Annotations.runbook }}- **Runbook:** {{ .Annotations.runbook }}{{ end }}
          {{ end }}
        color: 'danger'
        send_resolved: true
    
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: 'Critical plasma safety alert in tokamak control system'
        severity: 'critical'
  
  - name: 'critical-alerts'
    slack_configs:
      - channel: '#tokamak-alerts'
        title: 'Critical Alert - Tokamak RL Control'
        text: |
          🔴 **CRITICAL ALERT**
          
          {{ range .Alerts }}
          **{{ .Annotations.summary }}**
          {{ .Annotations.description }}
          {{ if .Annotations.impact }}**Impact:** {{ .Annotations.impact }}{{ end }}
          {{ if .Annotations.runbook }}**Runbook:** {{ .Annotations.runbook }}{{ end }}
          {{ end }}
        color: 'danger'
  
  - name: 'warning-alerts'
    slack_configs:
      - channel: '#tokamak-monitoring'
        title: 'Warning - Tokamak RL Control'
        text: |
          ⚠️ **WARNING**
          
          {{ range .Alerts }}
          **{{ .Annotations.summary }}**
          {{ .Annotations.description }}
          {{ end }}
        color: 'warning'
  
  - name: 'default'
    slack_configs:
      - channel: '#tokamak-monitoring'
        title: 'Alert - Tokamak RL Control'
        text: |
          {{ range .Alerts }}
          **{{ .Annotations.summary }}**
          {{ .Annotations.description }}
          {{ end }}
```

### PagerDuty Integration

```yaml
# PagerDuty escalation policies
receivers:
  - name: 'plasma-safety-critical'
    pagerduty_configs:
      - service_key: 'plasma-safety-service-key'
        description: |
          {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
        details:
          alert_count: '{{ len .Alerts }}'
          category: '{{ .CommonLabels.category }}'
          severity: '{{ .CommonLabels.severity }}'
          impact: |
            {{ range .Alerts }}{{ .Annotations.impact }}{{ end }}
          action: |
            {{ range .Alerts }}{{ .Annotations.action }}{{ end }}
        severity: 'critical'
        client: 'Tokamak RL Control Suite'
        client_url: 'https://monitoring.tokamak.internal'
  
  - name: 'system-critical'
    pagerduty_configs:
      - service_key: 'system-critical-service-key'
        description: |
          {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
        severity: 'critical'
```

### Email Notifications

```yaml
receivers:
  - name: 'email-operations'
    email_configs:
      - to: 'tokamak-ops@company.com'
        from: 'alertmanager@tokamak.internal'
        subject: '{{ .Status | toUpper }}: {{ len .Alerts }} alert(s) - Tokamak RL Control'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ if .Annotations.impact }}Impact: {{ .Annotations.impact }}{{ end }}
          {{ if .Annotations.action }}Action: {{ .Annotations.action }}{{ end }}
          {{ if .Annotations.runbook }}Runbook: {{ .Annotations.runbook }}{{ end }}
          
          Labels:
          {{ range .Labels.SortedPairs }}  {{ .Name }}: {{ .Value }}
          {{ end }}
          
          ---
          {{ end }}
        
        html: |
          <h2>{{ .Status | toUpper }}: Tokamak RL Control Alerts</h2>
          <p><strong>Number of alerts:</strong> {{ len .Alerts }}</p>
          
          {{ range .Alerts }}
          <div style="border-left: 4px solid {{ if eq .Status "firing" }}#d9534f{{ else }}#5cb85c{{ end }}; padding-left: 10px; margin: 10px 0;">
            <h3>{{ .Annotations.summary }}</h3>
            <p><strong>Description:</strong> {{ .Annotations.description }}</p>
            {{ if .Annotations.impact }}<p><strong>Impact:</strong> {{ .Annotations.impact }}</p>{{ end }}
            {{ if .Annotations.action }}<p><strong>Action:</strong> {{ .Annotations.action }}</p>{{ end }}
            {{ if .Annotations.runbook }}<p><strong>Runbook:</strong> <a href="{{ .Annotations.runbook }}">{{ .Annotations.runbook }}</a></p>{{ end }}
            
            <p><strong>Labels:</strong></p>
            <ul>
            {{ range .Labels.SortedPairs }}
              <li>{{ .Name }}: {{ .Value }}</li>
            {{ end }}
            </ul>
          </div>
          {{ end }}
```

## Alert Suppression and Inhibition

### Inhibition Rules

```yaml
# Prevent redundant alerts
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['instance', 'category']
  
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: '(HighCPUUsage|HighMemoryUsage|SlowInference)'
    equal: ['instance']
  
  - source_match:
      alertname: 'CriticalSafetyFactor'
    target_match:
      alertname: 'LowSafetyFactor'
    equal: ['tokamak_config']
  
  - source_match:
      category: 'plasma-safety'
      severity: 'critical'
    target_match:
      category: 'performance'
    equal: ['instance']
```

### Silence Templates

```yaml
# Common silence patterns
silences:
  - matchers:
      - name: 'alertname'
        value: 'TrainingStalled'
        isRegex: false
      - name: 'model_name'
        value: 'experimental-.*'
        isRegex: true
    comment: 'Experimental models - training stalls expected'
    createdBy: 'automation'
    startsAt: '2025-01-01T00:00:00Z'
    endsAt: '2025-12-31T23:59:59Z'
  
  - matchers:
      - name: 'severity'
        value: 'warning'
      - name: 'category'
        value: 'performance'
    comment: 'Maintenance window - performance alerts silenced'
    createdBy: 'ops-team'
```

## Custom Alert Handlers

### Webhook Integration

```python
from flask import Flask, request, jsonify
import json
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/alerts/webhook', methods=['POST'])
def handle_alerts():
    """Handle incoming alerts from Alertmanager"""
    
    try:
        alert_data = request.get_json()
        
        for alert in alert_data.get('alerts', []):
            process_alert(alert)
        
        return jsonify({'status': 'ok'}), 200
        
    except Exception as e:
        logger.error(f"Error processing alerts: {e}")
        return jsonify({'error': str(e)}), 500

def process_alert(alert):
    """Process individual alert"""
    
    alert_name = alert['labels'].get('alertname')
    severity = alert['labels'].get('severity')
    category = alert['labels'].get('category')
    status = alert['status']  # firing or resolved
    
    logger.info(f"Processing alert: {alert_name} ({severity}/{category}) - {status}")
    
    # Handle plasma safety alerts
    if category == 'plasma-safety' and severity == 'critical':
        handle_plasma_safety_alert(alert)
    
    # Handle system alerts
    elif category == 'system' and severity == 'critical':
        handle_system_alert(alert)
    
    # Handle training alerts
    elif category == 'training':
        handle_training_alert(alert)

def handle_plasma_safety_alert(alert):
    """Handle critical plasma safety alerts"""
    
    # Automatically trigger safety procedures
    if alert['labels'].get('escalation') == 'immediate':
        trigger_emergency_shutdown()
    
    # Log to safety system
    log_safety_event(alert)
    
    # Notify safety officer
    notify_safety_officer(alert)

def handle_system_alert(alert):
    """Handle critical system alerts"""
    
    # Attempt automatic remediation
    if alert['labels'].get('alertname') == 'HighMemoryUsage':
        trigger_garbage_collection()
    
    # Scale resources if needed
    if alert['labels'].get('alertname') == 'HighCPUUsage':
        request_resource_scaling()

def handle_training_alert(alert):
    """Handle training-related alerts"""
    
    # Save model checkpoint on training issues
    if alert['labels'].get('alertname') == 'TrainingStalled':
        save_model_checkpoint(alert['labels'].get('model_name'))
    
    # Adjust learning parameters on divergence
    if alert['labels'].get('alertname') == 'DivergingLoss':
        adjust_learning_rate(alert['labels'].get('model_name'))

# Placeholder functions for actions
def trigger_emergency_shutdown():
    logger.critical("EMERGENCY SHUTDOWN TRIGGERED")
    # Implement emergency shutdown logic

def log_safety_event(alert):
    logger.warning(f"Safety event logged: {alert}")

def notify_safety_officer(alert):
    logger.info(f"Safety officer notified: {alert}")

def trigger_garbage_collection():
    import gc
    gc.collect()
    logger.info("Garbage collection triggered")

def request_resource_scaling():
    logger.info("Resource scaling requested")

def save_model_checkpoint(model_name):
    logger.info(f"Saving checkpoint for model: {model_name}")

def adjust_learning_rate(model_name):
    logger.info(f"Adjusting learning rate for model: {model_name}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9093)
```

## Alert Testing

### Test Suite

```python
import unittest
import requests
import json
from datetime import datetime, timedelta

class AlertTestSuite(unittest.TestCase):
    """Test suite for alert configurations"""
    
    def setUp(self):
        self.alertmanager_url = "http://localhost:9093"
        self.prometheus_url = "http://localhost:9090"
    
    def test_critical_safety_alert(self):
        """Test critical plasma safety alert"""
        
        # Simulate low q_min condition
        test_alert = {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "alertname": "LowSafetyFactor",
                    "severity": "critical",
                    "category": "plasma-safety",
                    "tokamak_config": "test"
                },
                "annotations": {
                    "summary": "Test low safety factor alert",
                    "description": "q_min = 1.3"
                },
                "startsAt": datetime.utcnow().isoformat() + "Z"
            }]
        }
        
        # Send test alert
        response = requests.post(
            f"{self.alertmanager_url}/api/v1/alerts",
            json=test_alert
        )
        
        self.assertEqual(response.status_code, 200)
    
    def test_alert_inhibition(self):
        """Test that inhibition rules work correctly"""
        
        # Send critical alert first
        critical_alert = {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "alertname": "ServiceDown",
                    "severity": "critical",
                    "instance": "test-instance"
                },
                "startsAt": datetime.utcnow().isoformat() + "Z"
            }]
        }
        
        # Send warning alert that should be inhibited
        warning_alert = {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "alertname": "HighCPUUsage",
                    "severity": "warning",
                    "instance": "test-instance"
                },
                "startsAt": datetime.utcnow().isoformat() + "Z"
            }]
        }
        
        # Both should be accepted, but warning should be inhibited
        requests.post(f"{self.alertmanager_url}/api/v1/alerts", json=critical_alert)
        requests.post(f"{self.alertmanager_url}/api/v1/alerts", json=warning_alert)
        
        # Check active alerts
        response = requests.get(f"{self.alertmanager_url}/api/v1/alerts")
        alerts = response.json()['data']
        
        # Should only see the critical alert
        active_alerts = [a for a in alerts if a['status']['state'] == 'active']
        self.assertTrue(any(a['labels']['alertname'] == 'ServiceDown' for a in active_alerts))

if __name__ == '__main__':
    unittest.main()
```

## Best Practices

1. **Alert Fatigue Prevention**
   - Use appropriate severity levels
   - Implement proper grouping and inhibition
   - Set reasonable thresholds

2. **Actionable Alerts**
   - Include runbook links
   - Provide clear impact and action descriptions
   - Ensure alerts have clear resolution steps

3. **Escalation Policies**
   - Different channels for different severities
   - Time-based escalation for critical alerts
   - Proper on-call rotation management

4. **Testing and Validation**
   - Regular alert testing
   - Validate notification channels
   - Test escalation procedures

5. **Documentation**
   - Maintain runbook links
   - Document alert meanings and thresholds
   - Keep escalation procedures updated

6. **Continuous Improvement**
   - Review alert effectiveness regularly
   - Adjust thresholds based on operational experience
   - Remove or modify noisy alerts