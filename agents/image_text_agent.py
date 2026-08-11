"""
Image+Text Agent — Gemini multimodal analysis (conditional).
Only invoked when the user provides an actual image path.
Costs 1 Gemini API call per case.
"""
import os

import settings


class ImageTextAgent:
    """Tool: Analyses an uploaded image against the claimed return reason."""

    def __init__(self):
        self.enabled = bool(settings.GEMINI_API_KEY)
        if not self.enabled:
            print("[ImageTextAgent] No GEMINI_API_KEY set -> agent disabled.")
            return

        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        print("[ImageTextAgent] Ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_path: str, case_data: dict,
            custom_reason: str = None) -> dict:
        """
        Analyse the image for consistency with the return claim.

        Returns dict with: consistency_score, red_flags, assessment, skipped
        """
        if not self.enabled:
            return {'error': 'GEMINI_API_KEY not configured', 'skipped': True}

        if not os.path.exists(image_path):
            return {'error': f'Image not found: {image_path}', 'skipped': True}

        import PIL.Image
        img = PIL.Image.open(image_path)
        raw = case_data['raw']
        reason_text = custom_reason or raw.get('reason_category', 'unknown')

        prompt = (
            "You are a return fraud analyst for Flipkart. "
            "A customer is requesting a return for this product:\n\n"
            f"- Category: {raw.get('category', 'N/A')}\n"
            f"- Product Price: ₹{raw.get('price', 0):,.0f}\n"
            f"- Order Value: ₹{raw.get('order_value', 0):,.0f}\n"
            f"- Return Reason: {reason_text}\n"
            f"- Return Type: {raw.get('return_type', 'N/A')}\n\n"
            "The customer uploaded this image as evidence.\n\n"
            "Assess the image and respond in EXACTLY this format:\n"
            "CONSISTENCY: [1-5]  (1=clearly inconsistent, 5=fully supports claim)\n"
            "RED_FLAGS: [comma-separated list, or 'none']\n"
            "ASSESSMENT: [1–3 sentence assessment]\n"
        )

        try:
            response = self.model.generate_content([img, prompt])
            text = response.text
        except Exception as e:
            return {'error': str(e), 'skipped': True}

        # ── Parse structured response ─────────────────────────────────────────
        consistency = 3
        red_flags: list[str] = []
        assessment = text

        for line in text.split('\n'):
            line_s = line.strip()
            if line_s.upper().startswith('CONSISTENCY:'):
                try:
                    consistency = int(line_s.split(':')[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line_s.upper().startswith('RED_FLAGS:'):
                flags = line_s.split(':', 1)[1].strip()
                if flags.lower() != 'none':
                    red_flags = [f.strip() for f in flags.split(',') if f.strip()]
            elif line_s.upper().startswith('ASSESSMENT:'):
                assessment = line_s.split(':', 1)[1].strip()

        return {
            'consistency_score': consistency,
            'red_flags': red_flags,
            'assessment': assessment,
            'raw_response': text,
            'skipped': False,
        }
