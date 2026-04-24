"""
Email Notification System
=========================
Sends HTML-formatted fault-alert emails to maintenance teams when the
predictive model detects a fault.

Features:
  - Severity-coded colour badges (green / amber / red)
  - Retry logic with exponential back-off
  - Configurable from config.yaml
"""

import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


SEVERITY_COLOURS = {
    "low": "#FFA500",     # amber
    "high": "#FF3333",    # red
    "combined": "#CC0000",# dark red
    "normal": "#33CC33",  # green
}


def _severity_badge(fault_class: str, severity: float) -> tuple[str, str]:
    """Return (colour_hex, severity_label)."""
    if "severe" in fault_class or "combined" in fault_class:
        return SEVERITY_COLOURS["high"], "HIGH"
    elif "early" in fault_class:
        return SEVERITY_COLOURS["low"], "EARLY WARNING"
    else:
        return SEVERITY_COLOURS["normal"], "NORMAL"


def build_html_email(
    fault_class: str,
    confidence: float,
    severity: float,
    machine_id: str,
    sensor_location: str,
    timestamp: str | None = None,
) -> str:
    """Build a rich HTML email body."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    colour, sev_label = _severity_badge(fault_class, severity)

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f8; padding: 20px;">
      <div style="max-width: 600px; margin: auto; background: #ffffff; border-radius: 12px;
                  box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden;">
        <!-- header -->
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 24px 32px;">
          <h1 style="color: #fff; margin: 0; font-size: 22px;">
            ⚠️ Predictive Maintenance Alert
          </h1>
          <p style="color: #a0a0c0; margin: 6px 0 0 0; font-size: 13px;">
            Caterpillar Inc. — Health Monitoring System
          </p>
        </div>

        <!-- severity badge -->
        <div style="padding: 24px 32px 0;">
          <span style="display: inline-block; background: {colour}; color: #fff;
                       padding: 6px 18px; border-radius: 20px; font-weight: 700;
                       font-size: 14px; letter-spacing: 1px;">
            {sev_label}
          </span>
        </div>

        <!-- details -->
        <div style="padding: 20px 32px;">
          <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <tr><td style="padding: 8px 0; color: #888;">Fault Type</td>
                <td style="padding: 8px 0; font-weight: 600;">{fault_class.replace('_', ' ').title()}</td></tr>
            <tr><td style="padding: 8px 0; color: #888;">Confidence</td>
                <td style="padding: 8px 0; font-weight: 600;">{confidence:.1%}</td></tr>
            <tr><td style="padding: 8px 0; color: #888;">Severity</td>
                <td style="padding: 8px 0; font-weight: 600;">{severity:.2f} / 1.00</td></tr>
            <tr><td style="padding: 8px 0; color: #888;">Machine</td>
                <td style="padding: 8px 0; font-weight: 600;">{machine_id}</td></tr>
            <tr><td style="padding: 8px 0; color: #888;">Sensor Location</td>
                <td style="padding: 8px 0; font-weight: 600;">{sensor_location.replace('_', ' ').title()}</td></tr>
            <tr><td style="padding: 8px 0; color: #888;">Detected At</td>
                <td style="padding: 8px 0; font-weight: 600;">{timestamp}</td></tr>
          </table>
        </div>

        <!-- footer -->
        <div style="background: #f8f8fc; padding: 16px 32px; font-size: 12px; color: #999;
                    border-top: 1px solid #eee;">
          This is an automated alert from the Predictive Maintenance System.
          Please inspect the equipment at the earliest convenience.
        </div>
      </div>
    </body>
    </html>
    """
    return html


def send_alert(
    fault_class: str,
    confidence: float,
    severity: float,
    machine_id: str,
    sensor_location: str,
    email_cfg: dict | None = None,
    max_retries: int = 3,
) -> bool:
    """
    Send a fault-alert email. Returns True on success.

    If email_cfg is None, loads from config.yaml.
    """
    if email_cfg is None:
        cfg = load_config()
        email_cfg = cfg["email"]

    subject = (
        f"{email_cfg['subject_prefix']} {fault_class.replace('_', ' ').title()} "
        f"— {machine_id} @ {sensor_location}"
    )
    body = build_html_email(fault_class, confidence, severity, machine_id, sensor_location)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_cfg["from_address"]
    msg["To"] = ", ".join(email_cfg["recipients"])
    msg.attach(MIMEText(body, "html"))

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"]) as server:
                if email_cfg.get("use_tls", True):
                    server.starttls()
                server.login(email_cfg["username"], email_cfg["password"])
                server.sendmail(
                    email_cfg["from_address"],
                    email_cfg["recipients"],
                    msg.as_string(),
                )
            print(f"[Notifier] Alert sent successfully (attempt {attempt})")
            return True
        except Exception as e:
            wait = 2 ** attempt
            print(f"[Notifier] Attempt {attempt} failed: {e}. Retrying in {wait}s ...")
            time.sleep(wait)

    print("[Notifier] All retry attempts exhausted — email NOT sent.")
    return False


if __name__ == "__main__":
    # dry-run: build email HTML and print (no actual send)
    html = build_html_email(
        fault_class="bearing_wear_severe",
        confidence=0.94,
        severity=0.82,
        machine_id="CAT-EX-001",
        sensor_location="bearing_left",
    )
    print(html)
