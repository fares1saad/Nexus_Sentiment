# Sentiment Analysis Flask Application for Nexus Platform

> This repository contains a **Flask-based Sentiment Analysis API** developed for the **Nexus mental health platform**. Nexus is a web application that enables users to interact through social forums, complete mental health assessments, and book appointments with mental health professionals. The platform also provides a dedicated space for doctors, where they can be booked by users, view forum discussions, and offer professional advice.


🔗 **Nexus Platform Repository**
[https://github.com/AlaaEbrahim0/mental-mediator-api-graduation-project](https://github.com/AlaaEbrahim0/mental-mediator-api-graduation-project)

---

## Project Overview

The API provides **sentiment and mental health analysis models** that are integrated into the Nexus website. These models support content moderation, emotional understanding, and mental health assessment through machine learning–based sentiment analysis.

The system includes **two main models**, each serving a different purpose within the platform.

---

## Models Description

### 1. Hate Speech & Forum Sentiment Analysis Model

* Used in **Nexus social forums**
* Detects:

  * Hate speech
  * Toxic or harmful language
  * Overall sentiment of user-generated posts
* Helps maintain a **safe and supportive environment** for users

![alt text](images/image.png)

---

### 2. Depression Sentiment Analysis Model

This model is used in the **depression assessment feature** on the Nexus website.
The assessment is divided into **three components**:

1. **Beck’s Depression Inventory (BDI)**

   * Standard psychological questionnaire for depression assessment

2. **General Sentiment Analysis**

   * Classifies user input into **positive** or **negative** sentiment

3. **Custom Depression Detection Model**

   * A machine learning model specifically trained to detect depression-related linguistic patterns

#### Decision-Level Fusion

The final depression assessment result is produced using **Decision-Level Fusion**, combining the outputs of:

* Beck’s Depression Inventory
* General sentiment analysis
* Custom depression model

This fusion approach improves **robustness and reliability** of the final prediction.

![alt text](images/image-1.png)
![alt text](images/image-2.png)
![alt text](images/image-3.png)
---

## System Architecture

* Flask REST API
* Multiple ML models integrated into a unified inference pipeline
* Decision-Level Fusion for mental health prediction
* Deployed as a backend service for the Nexus platform

*(Architecture diagram or screenshots can be added here)*

---

## Requirements

* **Python 3.8 or later**

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

* Run the Flask application to start the API
* The API endpoints are consumed by the Nexus website for:

  * Forum content analysis
  * Depression assessment and sentiment evaluation

---

## Contact

For any inquiries, feedback, or collaboration opportunities, feel free to contact:

📧 **Fares Saad**
Email: [faresaad74@gmail.com](mailto:faresaad74@gmail.com) 

