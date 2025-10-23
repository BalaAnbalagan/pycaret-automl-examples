# Anomaly Detection - Network Intrusion Detection

## Overview

This notebook demonstrates anomaly detection using PyCaret to automatically identify malicious network traffic patterns that could indicate cyberattacks, intrusions, or abnormal behavior in network systems.

## Problem Statement

Detect anomalous network traffic patterns (potential intrusions, attacks, or abnormal behavior) using unsupervised learning techniques on network flow characteristics without requiring labeled data.

## Dataset

- **Source**: Synthetic network traffic data (based on CIC-IDS-2017 patterns)
- **Rows**: 10,000 network flow records
- **Features**: 8 network characteristics
- **Anomaly Rate**: ~10% (typical for network intrusion scenarios)
- **Type**: Unsupervised learning (no labels required)

### Features

All features represent network flow characteristics:

| Feature | Description | Type |
|---------|-------------|------|
| duration | Connection duration (seconds) | Numerical |
| protocol_type | Protocol (TCP/UDP/ICMP) | Categorical |
| service | Network service (HTTP/FTP/SSH/etc.) | Categorical |
| src_bytes | Bytes sent from source | Numerical |
| dst_bytes | Bytes sent to destination | Numerical |
| flag | Connection flag (SF/S0/REJ/etc.) | Categorical |
| count | Connections to same host (2 sec window) | Numerical |
| srv_count | Connections to same service (2 sec window) | Numerical |

## What You'll Learn

1. **Unsupervised Anomaly Detection**: Identify outliers without labels
2. **Multiple Algorithms**: Compare Isolation Forest, LOF, One-Class SVM
3. **Anomaly Scoring**: Assign anomaly scores to each data point
4. **Feature Engineering**: Prepare network data for anomaly detection
5. **Threshold Selection**: Balance precision vs. recall for anomaly detection
6. **Visualization**: Plot anomalies in reduced dimensional space
7. **Interpretation**: Understand why certain patterns are flagged
8. **Production Deployment**: Save models for real-time monitoring

## PyCaret Features Demonstrated

- `setup()`: Initialize anomaly detection environment
- `create_model()`: Create multiple anomaly detection models
  - Isolation Forest (iforest): Tree-based ensemble method
  - Local Outlier Factor (lof): Density-based detection
  - One-Class SVM (svm): Boundary-based detection
- `assign_model()`: Add anomaly labels and scores to dataset
- `plot_model()`: Visualizations (t-SNE, UMAP dimensionality reduction)
- `predict_model()`: Detect anomalies in new data
- `save_model()` / `load_model()`: Model persistence for production

## Running the Notebook

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/pycaret-automl-examples/blob/main/anomaly-detection/network_intrusion_detection.ipynb)

### Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook network_intrusion_detection.ipynb
```

## Expected Results

The models achieve:
- Detection of ~10% anomalies (network attacks/intrusions)
- High precision (few false positives minimize alert fatigue)
- Reasonable recall (catch most actual intrusions)
- Interpretable anomaly scores for prioritization
- Real-time detection capability

## Business Value

- **Security Operations Centers (SOC)**: Automated threat detection, reduce analyst workload
- **Network Administrators**: Identify unusual traffic patterns, prevent breaches
- **Compliance**: Meet regulatory requirements for intrusion detection
- **Incident Response**: Early warning system for security incidents
- **Cost Savings**: Prevent data breaches ($millions in potential damages)

## Key Insights

### Why Anomaly Detection for Network Security?

1. **Unknown Threats**: Detects new/zero-day attacks not seen during training
2. **No Labels Needed**: Works without manually labeled attack data
3. **Adaptable**: Learns normal behavior, flags deviations
4. **Real-Time**: Fast inference for continuous monitoring
5. **Complementary**: Works alongside signature-based intrusion detection systems (IDS)

### Algorithm Comparison

| Algorithm | Strengths | Best For |
|-----------|-----------|----------|
| **Isolation Forest** | Fast, handles high dimensions, good for global outliers | Large-scale network monitoring |
| **Local Outlier Factor** | Detects local anomalies, density-based | Contextual anomalies |
| **One-Class SVM** | Strong theoretical foundation, works well with kernel trick | Well-defined normal behavior |

### Anomaly Types Detected

1. **Port Scanning**: High connection counts to multiple ports
2. **DDoS Attacks**: Unusual traffic volume patterns
3. **Malware Communication**: Abnormal protocol usage
4. **Data Exfiltration**: Large outbound data transfers
5. **Brute Force Attacks**: Repeated failed connection attempts

## Deployment Applications

1. **Real-Time Monitoring**: Integrate with network monitoring systems (Splunk, ELK stack)
2. **SIEM Integration**: Feed anomaly alerts to Security Information and Event Management systems
3. **Automated Response**: Trigger firewall rules or quarantine suspicious hosts
4. **Forensics**: Retrospective analysis of historical network logs
5. **Threat Hunting**: Proactive search for indicators of compromise (IOCs)

## Performance Optimization

### Precision vs. Recall Trade-off

- **High Precision (Low FP)**: Set conservative threshold (anomaly_score > 0.7) to reduce alert fatigue
- **High Recall (Catch all attacks)**: Set aggressive threshold (anomaly_score > 0.3) to minimize missed threats
- **Balanced**: Use domain knowledge and business requirements to tune threshold

### Handling Imbalanced Data

- Network data is naturally imbalanced (normal >> anomalies)
- Anomaly detection handles this inherently (unsupervised)
- No need for oversampling/undersampling techniques

## Notebook Structure

Each code cell includes ELI20 explanations covering:
- **What**: Purpose of the code
- **Why**: Importance for anomaly detection
- **Technical Details**: How algorithms work
- **Expected Output**: What results to expect and how to interpret

## Prerequisites

- Python 3.8+
- Understanding of unsupervised learning concepts
- Basic knowledge of network protocols (helpful but not required)
- Familiarity with cybersecurity concepts (helpful)

## Time to Complete

Approximately 30-40 minutes

## Limitations and Considerations

1. **False Positives**: Normal but unusual behavior may trigger alerts (e.g., legitimate spike in traffic)
2. **Concept Drift**: Network patterns change over time, models need periodic retraining
3. **Feature Engineering**: Model quality heavily depends on meaningful features
4. **Threshold Tuning**: Requires domain expertise and business context
5. **Complementary Approach**: Should be used alongside signature-based detection, not as replacement

## Author

**Bala Anbalagan**

## Disclaimer

This model is for educational purposes and demonstrates anomaly detection techniques. Production deployment requires:
- Integration with enterprise security infrastructure
- Tuning for specific network environment
- Human-in-the-loop validation
- Compliance with security policies and regulations
- Regular model updates and monitoring
