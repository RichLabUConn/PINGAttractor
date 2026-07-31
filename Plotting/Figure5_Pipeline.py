
import subprocess
import sys

# =========================
# Toggle parallel processing
# =========================


def run_pipeline():
    print("Starting pipeline...\n")

    # =========================
    # Choose processing script
    # =========================
    processing_script = "Figure5_Heatmap.py"

    # =========================
    # Step 1: CSV Processing
    # =========================
    print(f"Running {processing_script}...")
    subprocess.run(
        [sys.executable, processing_script],
        check=True
    )

    # =========================
    # Step 2: Heatmap Plotting
    # =========================
    print("\nRunning Figure5_Heatmap.py...")
    subprocess.run(
        [sys.executable, "Figure5_Heatmap.py"],
        check=True
    )

    print("\n[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()