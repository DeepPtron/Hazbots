LoRa Bot Monitor System

A real-time multi-bot monitoring and human detection system designed for disaster response scenarios. The system enables tracking of multiple underground bots equipped with environmental sensors and LoRa communication, with AI-powered human voice detection capabilities.

## Features

### Multi-Bot Tracking
- **Real-time Monitoring**: Track multiple bots simultaneously with live sensor data updates
- **WebSocket Integration**: Instant data push to connected dashboards
- **Auto-tagging System**: Each bot gets a unique tag (#01, #02, etc.) for easy identification
- **GPS Tracking**: Monitor bot locations on an interactive map

### Environmental Sensing
- **Temperature Monitoring**: Real-time temperature readings
- **Pressure & Altitude**: Atmospheric pressure and calculated altitude
- **Gas Detection**: CO (Carbon Monoxide) concentration in PPM
- **Acceleration Data**: Movement and orientation sensing
- **Audio Levels**: Ambient sound monitoring (RMS values)

### Human Detection (AI-Powered)
- **Silero VAD**: Deep learning-based Voice Activity Detection
- **Multi-format Support**: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS
- **Confidence Scoring**: Percentage-based detection confidence
- **Energy Analysis**: Audio loudness and activity detection
- **Real-time Alerts**: Instant notification when human voice is detected

### 📊 Data Export
- **XLSX Export**: Download all bot data as Excel spreadsheet
- **Historical Data**: Maintains up to 500 data points per bot
- **Timestamped Records**: All readings include precise timestamps

---

## 📁 Project Structure

```
karan haz bots copy/
├── backend.py           # Flask server with WebSocket & APIs
├── simulate_bots.py     # Bot simulator for testing
├── lora-bot-monitor.html# Dashboard frontend
├── aiml.py              # Folder-based human detection
├── human_detector.py    # File-picker human detection
├── mix_audio.py         # Audio mixing utility
├── requirements.txt     # Python dependencies
├── arduino file.cpp     #arduino file
└── README.md           # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone/Download the project**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install flask flask-cors flask-socketio openpyxl
   ```

4. **Run the server**
   ```bash
   python backend.py
   ```

5. **Open the dashboard**
   - Navigate to `http://localhost:5000` in your browser

6. **(Optional) Run the bot simulator**
   ```bash
   python simulate_bots.py
   ```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the monitoring dashboard |
| `/api/data` | POST | Receive sensor data from bots |
| `/api/bots` | GET | Get list of all registered bots |
| `/api/latest` | GET | Get latest data for all/specific bot |
| `/api/detect-human` | POST | Upload audio file for human detection |
| `/api/export` | GET | Export all data to XLSX file |

---

## 🔧 How It Works

1. **Data Collection**: ESP32-based bots collect environmental sensor data and transmit via LoRa
2. **Backend Processing**: Flask server receives, processes, and stores bot data
3. **Real-time Updates**: WebSocket pushes updates to all connected dashboards
4. **Human Detection**: Uploaded audio is analyzed using Silero VAD deep learning model
5. **Visualization**: Interactive dashboard displays bot locations, sensor readings, and alerts

---

## 🔮 Future Plans

### Phase 1: Enhanced Detection
- [ ] Improve human detection accuracy with additional ML models
- [ ] Add distress signal recognition (whistles, shouts, tapping patterns)
- [ ] Implement noise filtering for underground environments

### Phase 2: Advanced Analytics
- [ ] Predictive analytics for bot battery life and signal quality
- [ ] Heat mapping of detected human presence
- [ ] Historical trend analysis and reporting

### Phase 3: Hardware Integration
- [ ] Direct ESP32 LoRa firmware integration
- [ ] Support for additional sensor types (humidity, vibration)
- [ ] Mesh networking between bots for extended range

### Phase 4: Rescue Operations
- [ ] Integration with emergency response systems
- [ ] Automated priority alerts based on detection confidence
- [ ] Path planning for rescue teams based on bot positions
- [ ] Two-way audio communication capabilities

### Phase 5: Scalability
- [ ] Cloud deployment with multi-region support
- [ ] Database integration for long-term data storage
- [ ] Mobile app for field responders
- [ ] API for third-party integrations

---

## 🛠️ Technologies Used

- **Backend**: Python, Flask, Flask-SocketIO
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: PyTorch, Silero VAD
- **Audio Processing**: Librosa, TorchAudio, SoundFile
- **Data Export**: OpenPyXL

---

## 📝 License

This project is developed for disaster response and humanitarian purposes.

---

## 👥 Contributors

- Hazbots Team

---

*Built with ❤️ for saving lives*
