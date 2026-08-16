# NephroScan AI — Demo Script

## Opening (30 seconds)

"NephroScan is an AI-assisted medical imaging screening prototype. It uses deep learning to classify medical images across four modalities: kidney stones, chest pneumonia, brain MRI, and heart cardiomegaly."

## Live Demo Flow

### 1. Dashboard Overview (1 minute)
- Show the dashboard with model status indicators
- Point out all 4 models loaded and ready
- Navigate through views to show the interface

### 2. Run an Analysis (2 minutes)
- Go to New Analysis
- Upload a demo image (kidney CT, chest X-ray, brain MRI, or heart X-ray)
- Select scan type and enter clinical indication
- Click Run Analysis
- Show the result with confidence score and provenance label
- Point out: "Every result shows the model used, timestamp, and inference type"

### 3. Grad-CAM Explainability (1 minute)
- Toggle Explainability Overlay on the result
- Show the attention heatmap
- Explain: "This shows where the model focused — it's not a diagnosis or segmentation"

### 4. Expo Presence Demo (2 minutes)
- Navigate to Expo Human Presence
- Click Start Camera
- Show the live optical view and thermal proxy side by side
- Point out: "The thermal channel is a visual simulation — not an infrared measurement"
- Show the presence log populating with detection events
- Stop the camera

### 5. NephroBot Assistant (1 minute)
- Open the assistant
- Ask "Explain my result"
- Show the AI-generated explanation with disclaimer
- Ask "What should I ask my doctor?"
- Show the suggested questions

## Safety Talking Points

- "This is an educational prototype, not a diagnostic device"
- "Every result includes provenance metadata for transparency"
- "The thermal proxy is software-generated — not a temperature measurement"
- "All results require clinical validation by a qualified professional"
- "No patient data is stored — everything stays in the browser session"

## Closing (30 seconds)

"NephroScan demonstrates how AI can assist medical imaging screening while maintaining transparency through provenance labels and explainability. The system is designed for deployment on standard web infrastructure with no specialized hardware."

## Troubleshooting

- **Camera not working:** Ensure HTTPS or localhost. Check browser permissions.
- **Models not loading:** Verify `models/` directory contains all 4 `.pth` files
- **CORS errors:** Set `CORS_ORIGINS` environment variable
- **Slow inference:** First run may be slower as models warm up
