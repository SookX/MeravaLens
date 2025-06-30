<!--
Hey, thanks for using the awesome-readme-template template.  
If you have any enhancements, then fork this project and create a pull request 
or just open an issue with the label "enhancement".

Don't forget to give this project a star for additional support ;)
Maybe you can mention me or this repo in the acknowledgements too
-->
<div align="center">
  <br>
  <img src="https://i.ibb.co/sdS5Mghy/meravalens-high-resolution-logo-grayscale-transparent.png" alt="logo" width="300" height="auto" />
  <br>
  <br>
  <br>
  

  
  
<!-- Badges -->
<p>
  <a href="https://github.com/SookX/MeravaLens/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/SookX/MeravaLens" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/SookX/MeravaLens" alt="last update" />
  </a>
  <a href="https://github.com/SookX/MeravaLens/network/members">
    <img src="https://img.shields.io/github/forks/SookX/MeravaLens" alt="forks" />
  </a>
  <a href="https://github.com/SookX/MeravaLens/stargazers">
    <img src="https://img.shields.io/github/stars/SookX/MeravaLens" alt="stars" />
  </a>
  <a href="https://github.com/SookX/MeravaLens/issues/">
    <img src="https://img.shields.io/github/issues/SookX/MeravaLens" alt="open issues" />
  </a>
  <a href="https://github.com/SookX/MeravaLens/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/SookX/MeravaLens.svg" alt="license" />
  </a>
</p>
   

</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
  * [Screenshots](#camera-screenshots)
  * [Tech Stack](#space_invader-tech-stack)
  * [Features](#dart-features)
  * [Environment Variables](#key-environment-variables)
- [Getting Started](#toolbox-getting-started)
  * [Installation](#gear-installation)
  * [Run Locally](#running-run-locally)
  * [Deployment](#triangular_flag_on_post-deployment)
- [Contributing](#wave-contributing)
- [License](#warning-license)

<!-- About the Project -->
## 🌍 About the Project

**MeravaLens** is an advanced **geospatial intelligence platform** designed to deliver **comprehensive, data-driven analysis** of Earth's surface.

It combines:

- 🛰️ Open-access **satellite imagery**
- 🌐 Robust **public APIs**
- 🤖 Proprietary **deep learning models**

These technologies work together to provide **accurate, real-time insights** into:

- Land use and cover classification  
- Infrastructure mapping  
- Environmental feature detection  
- Agricultural and urban monitoring

Whether you're a **researcher**, **urban planner**, **environmental analyst**, or **decision-maker**, MeravaLens equips you with the tools to extract meaningful intelligence from raw geospatial data — anywhere on the planet.
<!-- Screenshots -->
### :camera: Screenshots

<div align="center"> 
  <img src="https://i.ibb.co/fzd6867L/Screenshot-2025-06-30-140342.png" width="600" alt="screenshot" />
</div>


<!-- TechStack -->
### :space_invader: Tech Stack
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)  
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)  
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)  
![MUI](https://img.shields.io/badge/MUI-007FFF?style=for-the-badge&logo=mui&logoColor=white)  
![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)  
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)



<!-- Features -->
## 🎯 Features

MeravaLens offers a robust set of features that enable end-to-end geospatial analysis powered by modern web technologies and deep learning:

### 🛰️ 1. Satellite Imagery Access
- Seamless integration with **Google Maps** Static API
- Retrieve **high-resolution satellite images** of any location globally
- Supports dynamic zoom and customizable map parameters

### 🌦️ 2. Public Environmental Data APIs
- **[Current Weather Data API](https://openweathermap.org/current)**  
  Retrieves real-time weather data for any location by geographic coordinates. Includes temperature, humidity, wind speed, and more.

- **[Air Pollution API](https://openweathermap.org/api/air-pollution)**  
  Provides air quality information based on pollutants such as PM2.5, PM10, CO, SO₂, NO₂, and O₃.

Both APIs are used to add environmental context to the geographic location being analyzed, enhancing the interpretability of satellite imagery and land use statistics.

### 🧠 3. AI-Powered Semantic Segmentation

MeravaLens integrates a custom-built **ResUNet** deep learning model to perform high-resolution semantic segmentation on satellite imagery. This model enables detailed pixel-level classification of land cover types such as **buildings**, **roads**, **forests**, **agriculture**, and more.

#### 🛠️ Model Architecture

- **Encoder:** Utilizes a pretrained `ResNet-18` backbone for efficient and accurate feature extraction, benefiting from deep residual learning.
- **Decoder:** A custom convolutional decoder specifically designed to mirror the ResNet-18 architecture, allowing for effective **skip connections** between encoder and decoder layers to preserve spatial details.
- **Output:** Produces a multi-channel segmentation mask where each pixel is classified into one of the semantic land cover categories.

> 🧩 *Below is a visualization of the model's learned semantic understanding across different land cover types:*

<img src="https://i.ibb.co/27cz99Rz/Figure-3.png" width="800">

#### 📚 Dataset — LoveDA

The model is trained and evaluated on the **LoveDA (Land-cOver Domain Adaptation)** dataset, which is a comprehensive benchmark tailored for semantic segmentation in remote sensing.

**Key Features of LoveDA:**

- 📷 Over **19,000** high-resolution satellite images.
- 🏷️ Pixel-level annotations for **7 semantic classes**:
  - Background
  - Building
  - Road
  - Water
  - Barren
  - Forest
  - Agriculture
- 🌍 Designed to test generalization between **urban** and **rural** domain shifts.

> The use of LoveDA ensures strong generalization performance across diverse landscapes and enhances the robustness of semantic predictions in real-world Earth observation scenarios.

---

### 📖 References

- **LoveDA Dataset:**  
Wang, Y., Yang, M., Zhang, S., Chen, J., Wang, X., & Guo, Q. (2021). *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptation.* IEEE Transactions on Geoscience and Remote Sensing.  
[https://doi.org/10.1109/TGRS.2021.3104012](https://doi.org/10.1109/TGRS.2021.3104012)

- **U-Net Architecture:**  
Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* Medical Image Computing and Computer-Assisted Intervention (MICCAI).  
[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

- **ResNet Architecture:**  
He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

### 🔍 4. Smart Geospatial Insights via LLM

MeravaLens uses the output of its custom ResUNet semantic segmentation model as input for a powerful large language model to generate clear, detailed, and structured analysis of geographic areas. 

We utilize the **meta-llama/Llama-3.3-70B-Instruct** model from Hugging Face to translate pixel-level land cover data into human-readable insights, summarizing land use proportions, environmental features, and other relevant details.

You can explore the model here:  
[https://huggingface.co/meta-llama/Llama-3.3-70b-instruct](https://huggingface.co/meta-llama/Llama-3.3-70b-instruct)


<!-- Env Variables -->
## :key: Environment Variables

To run this project, you will need to add the following environment variables to your `.env` files.

---

### Server (Django)

```env
DJANGO_SECRET_KEY='your-django-secret-key'
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGIN=http://localhost:5173

DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
DB_HOST="your_db_host"
DB_PORT="your_db_port"

FRONTEND_URL=http://localhost:5173

EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your_email@example.com
EMAIL_HOST_PASSWORD=your_email_password
DEFAULT_FROM_EMAIL=your_email@example.com

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY='your-google-oauth2-client-id'

OPENWEATHER_API_KEY='your-openweather-api-key'

GOOGLE_API_KEY='your-google-api-key'
```

### Client (React)

```env
VITE_GOOGLE_OAUTH2=your-google-oauth2-client-id
VITE_GOOGLE_API_KEY="your-google-api-key"
```

### Microservice (FastAPI)

```env
HUGGING_FACE="your-huggingface-api-key"
```

<!-- Getting Started -->
## 	:toolbox: Getting Started


<!-- Installation -->

Install the dependencies for each part of the project separately.

---

```bash
  python -m venv venv
  source venv/bin/activate       # On Windows use: venv\Scripts\activate
  pip install -r requirements.txt
```

#### Client (React)

```bash
  cd client
  npm install
```

---


<!-- Run Locally -->
### :running: Run Locally

Clone the project

```bash
  git clone https://github.com/SookX/MeravaLens
```

Go to the project directory

```bash
  cd my-project
```

Start the client (React)

```bash
  cd client
  npm run dev
```

Start the server (Django)

```bash
  cd ../server
  python manage.py runserver
```

Start the microservice (FastAPI)

```bash
  cd ../ml/api
  uvicorn server:app --host 0.0.0.0 --port 80
```



<!-- Deployment -->
### :triangular_flag_on_post: Deployment

The model is deployed at:  
[your-deployment-url](https://meravalens-client.onrender.com/)



<!-- Contributing -->
## :wave: Contributing

<a href="https://github.com/SookX/MeravaLens/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SookX/MeravaLens" />
</a>



<!-- License -->
## :warning: License

Distributed under the no License. See LICENSE.txt for more information.



