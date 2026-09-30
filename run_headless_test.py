from cad_analyzer_final import CADAnalyzer
import json

openai_key = "YOUR_API_KEY"
gemini_key = "AIzaSyCEhfdP7w-PB3TvdLlXs7koi12AAUbOdYo"

analyzer = CADAnalyzer(openai_key, gemini_key)
json_out = analyzer.extract_cad_parameters('technical_drawing.jpg')

with open('final_submission.json', 'w') as f:
    json.dump(json_out, f, indent=2)
    
print("JSON extraction complete.")
