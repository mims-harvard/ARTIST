"""
Schemas exposed to the chat template. These tell the model what tools exist and
the expected JSON arguments. The *runtime* execution lives in tools_runtime.py
"""

TS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "timeseries_zoom_in_tool",
            "description": "Zoom in on a TS segment [x1, y1].",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_seg": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2, "maxItems": 2,
                    }
                },
                "required": ["ts_seg"]
            }
        }
    },

]
