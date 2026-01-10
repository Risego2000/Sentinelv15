# 🛡️ SENTINEL V15 - Traffic Enforcement HUD

**SENTINEL V15** is a high-performance, AI-driven traffic enforcement and forensic analysis system. It leverages TensorFlow.js for real-time edge detection and Google Gemini for deep forensic auditing of traffic infractions.

![SENTINEL HUD](https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6)

## 🚀 Key Features

- **Real-time Detection:** Optimized COCO-SSD pipeline with Advanced NMS and Class-Specific Thresholds.
- **Forensic Audit:** Direct integration with Gemini AI for detailed multi-spectral analysis of vehicle behavior, cell phone detection, and seatbelt compliance.
- **Dynamic Protocols:** Multi-lane detection system with pre-configured stacks for Urban, Rural, and High-Speed environments.
- **Edge Analytics:** Real-time speed estimation, trajectory tracking, and telemetry visualization.

## 🛠️ Tech Stack

- **Frontend:** React + Vite + Tailwind CSS
- **AI/ML:** TensorFlow.js + COCO-SSD
- **LLM Context:** Google Gemini API
- **Icons:** Lucide-React

## 📦 Deployment (Vercel)

The easiest way to deploy SENTINEL V15 is via [Vercel](https://vercel.com).

1. **Connect your GitHub Repo:** Select `Risego2000/SENTINELV15`.
2. **Configure Environment Variables:**
   - Go to Project Settings > Environment Variables.
   - Add `VITE_API_KEY`: Your Google Gemini API Key.
3. **Build Settings:**
   - Framework Preset: `Vite`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. **Deploy!**

## 💻 Local Setup

1. **Clone & Install:**
   ```bash
   git clone https://github.com/Risego2000/SENTINELV15.git
   cd SENTINELV15
   npm install
   ```
2. **API Key:**
   Create a `.env` file in the root:
   ```env
   VITE_API_KEY=your_gemini_api_key_here
   ```
3. **Run:**
   ```bash
   npm run dev
   ```

## 📜 Detection Protocols

- `Daganzon Protocol V15`: Comprehensive urban enforcement.
- `High-Speed Stack`: Advanced velocity and lane tracking.
- `School Zone Safety`: Extreme pedestrian priority and low-speed calibration.

---
Developed for **Daganzo de Arriba Local Police Force**.
