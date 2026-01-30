/*
 * ESP32-C3 HAZBOT v4.4 - WIFI DATA LINK
 * SENSORS: GPS | MQ7 | ADXL345 | BMP280 | INMP441
 * OUTPUT: WiFi (JSON HTTP POST) + ISD1820 (Local Alert)
 */

#include <Wire.h>
#include <SPI.h>
#include <TinyGPSPlus.h>
#include <LoRa.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_BMP280.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ==========================================
// WIFI & SERVER CONFIGURATION
// ==========================================
const char* WIFI_SSID = "LuI_0307";        // <--- ENTER WIFI NAME
const char* WIFI_PASSWORD = "42305030"; // <--- ENTER WIFI PASS
// Replace with the IP address of your backend server
const char* SERVER_URL = "http://10.250.170.39:5000/api/data";

// ==========================================
// PINOUT CONFIGURATION
// ==========================================
#define GPS_RX 19
#define GPS_TX 20
#define MQ7_PIN 1

// LoRa SX1278 (Kept for hardware init)
#define LORA_CS 7
#define LORA_SCK 4
#define LORA_MOSI 5
#define LORA_MISO 6
#define LORA_RST 10
#define LORA_DIO0 18
#define LORA_FREQ 433E6

// I2C Sensors (ADXL345 + BMP280)
#define I2C_SDA 3
#define I2C_SCL 2

// INMP441 I2S Microphone
#define I2S_WS 0        
#define I2S_SCK 8       
#define I2S_SD 9        
#define I2S_PORT I2S_NUM_0

// ISD1820 Voice Playback
#define ISD_PLAYE 21    

// ==========================================
// OBJECTS & VARIABLES
// ==========================================
TinyGPSPlus gps;
HardwareSerial GPS_SERIAL(2);
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(0x53);
Adafruit_BMP280 bmp;

// Data Variables
float co_ppm = 0.0f;
float accel_x = 0.0f, accel_y = 0.0f, accel_z = 0.0f;
float temperature = 0.0f;
float pressure_hpa = 0.0f;
float altitude_m = 0.0f;

// Audio Variables
float audio_rms = 0.0f;

// Status Flags
bool wifi_connected = false;
bool adxl_ok = false;
bool bmp_ok = false;
bool i2s_ok = false;
unsigned long last_cycle = 0;

// I2S Config
#define I2S_SAMPLE_RATE 16000
#define I2S_BUFFER_SIZE 512
int32_t i2s_read_buffer[I2S_BUFFER_SIZE];

// ==========================================
// WIFI FUNCTIONS
// ==========================================
void setup_wifi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 20) { // 10 sec timeout
        delay(500);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✅ WiFi Connected!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        wifi_connected = true;
    } else {
        Serial.println("\n❌ WiFi Connection Failed (Continuing offline)");
        wifi_connected = false;
    }
}

void send_data_to_server() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi Disconnected. Attempting reconnect...");
        WiFi.reconnect();
        return;
    }

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    // Construct JSON Payload manually to avoid external lib dependencies
    String jsonPayload = "{";
    jsonPayload += "\"device_id\": \"HAZBOT_01\",";
    
    // Environmental
    jsonPayload += "\"temperature\": " + String(temperature, 1) + ",";
    jsonPayload += "\"pressure\": " + String(pressure_hpa, 1) + ",";
    jsonPayload += "\"altitude\": " + String(altitude_m, 1) + ",";
    jsonPayload += "\"gas_co_ppm\": " + String(co_ppm, 1) + ",";
    
    // Motion
    jsonPayload += "\"accel_x\": " + String(accel_x, 2) + ",";
    jsonPayload += "\"accel_y\": " + String(accel_y, 2) + ",";
    jsonPayload += "\"accel_z\": " + String(accel_z, 2) + ",";
    
    // Audio Level (RMS)
    jsonPayload += "\"audio_rms\": " + String(audio_rms, 0) + ",";

    // GPS Data
    jsonPayload += "\"gps_valid\": " + String(gps.location.isValid() ? "true" : "false") + ",";
    if (gps.location.isValid()) {
        jsonPayload += "\"lat\": " + String(gps.location.lat(), 6) + ",";
        jsonPayload += "\"lng\": " + String(gps.location.lng(), 6);
    } else {
        jsonPayload += "\"lat\": 0,";
        jsonPayload += "\"lng\": 0";
    }
    
    jsonPayload += "}";

    // Send POST Request
    Serial.println("\n📤 Sending Data to Server...");
    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.printf("✅ Server Responded: %d\n", httpResponseCode);
        // Serial.println(response); // Uncomment to see server reply
    } else {
        Serial.printf("❌ Error Sending: %s\n", http.errorToString(httpResponseCode).c_str());
    }

    http.end();
}

// ==========================================
// SENSOR FUNCTIONS (Unchanged)
// ==========================================
void setup_gps() {
    Serial.print("GPS UART2... ");
    GPS_SERIAL.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);
    Serial.println("OK");
}

void read_gps() {
    while (GPS_SERIAL.available() > 0) {
        gps.encode(GPS_SERIAL.read());
    }
}

void setup_co_sensor() {
    Serial.print("MQ7 CO... ");
    pinMode(MQ7_PIN, INPUT);
    analogSetAttenuation(ADC_11db);
    Serial.println("OK");
}

void read_co_sensor() {
    int raw_value = analogRead(MQ7_PIN);
    co_ppm = (raw_value / 4095.0f) * 500.0f;
}

void setup_adxl345() {
    Serial.print("ADXL345... ");
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(100000);
    if (accel.begin()) {
        accel.setRange(ADXL345_RANGE_16_G);
        adxl_ok = true;
        Serial.println("OK");
    } else {
        Serial.println("FAILED");
    }
}

void read_adxl345() {
    if (!adxl_ok) return;
    sensors_event_t event;
    accel.getEvent(&event);
    accel_x = event.acceleration.x;
    accel_y = event.acceleration.y;
    accel_z = event.acceleration.z;
}

void setup_bmp280() {
    Serial.print("BMP280... ");
    if (bmp.begin(0x76) || bmp.begin(0x77)) {
        bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                        Adafruit_BMP280::SAMPLING_X2,
                        Adafruit_BMP280::SAMPLING_X16,
                        Adafruit_BMP280::FILTER_X16,
                        Adafruit_BMP280::STANDBY_MS_500);
        bmp_ok = true;
        Serial.println("OK");
    } else {
        Serial.println("FAILED");
    }
}

void read_bmp280() {
    if (!bmp_ok) return;
    temperature = bmp.readTemperature();
    pressure_hpa = bmp.readPressure() / 100.0f;
    altitude_m = bmp.readAltitude(1013.25);
}

// ==========================================
// AUDIO FUNCTIONS
// ==========================================
void setup_i2s_microphone() {
    Serial.print("INMP441 I2S... ");
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) == ESP_OK) {
        if (i2s_set_pin(I2S_PORT, &pin_config) == ESP_OK) {
            i2s_ok = true;
            Serial.println("OK");
        }
    }
}

void read_i2s_microphone() {
    if (!i2s_ok) return;
    size_t bytes_read = 0;
    i2s_read(I2S_PORT, &i2s_read_buffer, sizeof(i2s_read_buffer), &bytes_read, portMAX_DELAY);
    
    int samples_read = bytes_read / sizeof(int32_t);
    int64_t sum_squares = 0;
    for (int i = 0; i < samples_read; i++) {
        int32_t sample = i2s_read_buffer[i] >> 14;
        sum_squares += (int64_t)sample * sample;
    }
    if (samples_read > 0) {
        audio_rms = sqrt((float)sum_squares / samples_read);
    }
}

void setup_isd1820() {
    pinMode(ISD_PLAYE, OUTPUT);
    digitalWrite(ISD_PLAYE, HIGH);
}

void isd1820_playback() {
    Serial.println("🔊 Triggering Warning Playback");
    digitalWrite(ISD_PLAYE, LOW);
    delay(50);
    digitalWrite(ISD_PLAYE, HIGH);
}

// ==========================================
// MAIN SETUP & LOOP
// ==========================================
void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println("\n🌋 HAZBOT v4.4 - WIFI LINK");
    Serial.println("===========================");

    // Initialize Sensors
    setup_gps();
    setup_co_sensor();
    
    // LoRa hardware init (Just to keep pins stable, not sending)
    pinMode(LORA_RST, OUTPUT); digitalWrite(LORA_RST, HIGH);
    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);
    LoRa.setPins(LORA_CS, LORA_RST, LORA_DIO0);
    if(LoRa.begin(LORA_FREQ)) Serial.println("LoRa Standby... OK");

    setup_adxl345();
    setup_bmp280();
    setup_i2s_microphone();
    setup_isd1820();

    // Connect to WiFi
    setup_wifi();
}

void loop() {
    // 1. Read all sensors continuously
    read_gps();
    read_co_sensor();
    read_adxl345();
    read_bmp280();
    read_i2s_microphone();

    // 2. Local Logic: Trigger playback if audio is too loud (Landslide rumble)
    if (audio_rms > 5000.0f) {
        // Simple debounce could be added here
        // isd1820_playback(); // Uncomment to enable auto-playback
    }

    // 3. Manual Playback Trigger
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == 'P' || cmd == 'p') isd1820_playback();
    }

    // 4. Send Data via WiFi every 5 seconds
    if (millis() - last_cycle >= 5000UL) {
        last_cycle = millis();
        print_dashboard();      // Print to Serial Monitor
        send_data_to_server();  // Send JSON to Backend
    }
    
    delay(10);
}

void print_dashboard() {
    Serial.println("\n--- SENSOR STATUS ---");
    Serial.printf("Temp: %.1f C | Press: %.1f hPa\n", temperature, pressure_hpa);
    Serial.printf("CO: %.1f ppm | GPS: %s\n", co_ppm, gps.location.isValid() ? "LOCKED" : "SEARCHING");
    Serial.printf("Accel Z: %.2f | Audio RMS: %.0f\n", accel_z, audio_rms);
}
