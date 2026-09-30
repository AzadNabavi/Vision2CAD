import json
from multi_agent_council import MultiAgentCouncil

council = MultiAgentCouncil("technical_drawing.jpg")
prompt_10 = """
Based on the full 9-step council debate, generate the FINAL accurate JSON representing this part.
Output ONLY the JSON object.
"""
resp_10 = council.call_claude(prompt_10)
print("RAW RESPONSE:", repr(resp_10))
