"""
Portable Builder Module
=======================
Creates a standalone portable distribution of zone_coverage_viz using PyInstaller.

Architectural Overview:
    Responsibility: Build a self-contained executable bundle that can run on
    machines without Python installed.

    Key Interactions:
    - Input: zone_coverage_viz/ source code, Data/zone_coverage_data.json
    - Output: ZoneCoverageViz_Portable/ folder with .exe and launchers
    - Dependencies: PyInstaller (auto-installed if missing)

    Navigation Guide:
    - CONFIGURATION: Build settings and templates
    - BUILD FUNCTIONS: Core build logic
    - CLI ENTRY POINT: Command-line interface

Usage:
    # From command line
    python portable_builder.py

    # From Python
    from portable_builder import build_portable
    build_portable(output_dir="ZoneCoverageViz_Portable")

Requirements:
    pip install pyinstaller
"""

import subprocess
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 📋 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# MODIFICATION POINT: Add/remove hidden imports as needed for PyInstaller
HIDDEN_IMPORTS = [
    # === zone_coverage_viz package modules ===
    "zone_coverage_viz",
    "zone_coverage_viz.data_loader",
    "zone_coverage_viz.geometry_service",
    "zone_coverage_viz.viz_config_types",
    # === Core dependencies ===
    "flask",
    "flask_cors",
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    "geopandas",
    "pyproj",
    "pyproj.crs",
    "pyproj.datadir",
    "pyproj.database",
    "pyproj._datadir",
    "pyproj.transformer",
    "pandas",
    "pyogrio",
    "pyogrio._io",
    "pyogrio.raw",
    "multiprocessing",
]

# Modules to exclude from bundle (reduces size dramatically)
# MODIFICATION POINT: Aggressive exclusion of anaconda packages not needed
EXCLUDE_MODULES = [
    # === GUI / PLOTTING (NOT NEEDED - we use web browser) ===
    "matplotlib",
    "mpl_toolkits",
    "pylab",
    "tkinter",
    "_tkinter",
    "tcl",
    "tk",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "qtpy",
    "Qt",
    "sip",
    "PIL",
    "pillow",
    "Pillow",
    "pygame",
    "wx",
    "wxPython",
    "kivy",
    "pyglet",
    "vtk",
    "vtkmodules",  # VTK is massive
    "plotly",
    "plotly_express",
    "altair",
    "vega",
    "vegafusion",
    "bokeh",
    "seaborn",
    "holoviews",
    "hvplot",
    "datashader",
    "colorcet",
    # === SCIENTIFIC (NOT NEEDED - only using basic geopandas/shapely) ===
    "scipy",
    "sklearn",
    "scikit-learn",
    "statsmodels",
    "sympy",
    "networkx",
    "xarray",
    "netCDF4",
    "h5py",
    "hdf5",
    "tables",
    "pytables",
    "zarr",
    "dask",
    "distributed",
    "numba",
    "llvmlite",
    "cython",
    "Cython",
    # === JUPYTER / NOTEBOOKS (NOT NEEDED - running as server) ===
    "notebook",
    "jupyter",
    "jupyterlab",
    "IPython",
    "ipykernel",
    "ipywidgets",
    "nbformat",
    "nbconvert",
    "nbclient",
    "traitlets",
    "jupyter_client",
    "jupyter_core",
    "zmq",
    "pyzmq",
    # === DEV TOOLS (NOT NEEDED in production) ===
    "sphinx",
    "docutils",
    "numpydoc",
    "babel",
    # NOTE: gettext needed by Flask/click - don't exclude
    "jedi",
    "parso",
    "black",
    "isort",
    "autopep8",
    "yapf",
    "yapf_third_party",
    "pylint",
    "astroid",
    "flake8",
    "pyflakes",
    "pycodestyle",
    "mccabe",
    "mypy",
    "typing_extensions",
    "pytest",
    "coverage",
    "tox",
    "pdb",
    "pdbpp",
    "ipdb",
    # === DATA LOADING (NOT NEEDED - we use JSON not these formats) ===
    "xlrd",
    "xlwt",
    "openpyxl",
    "xlsxwriter",
    "odfpy",
    "pyxlsb",
    "sqlalchemy",
    "psycopg2",
    "pymysql",
    "sqlite3",
    "intake",
    "intake_parquet",
    "intake_xarray",
    "fastparquet",
    "pyarrow",
    "feather",
    "lxml",
    "html5lib",
    "beautifulsoup4",
    "bs4",
    # === MISC (NOT NEEDED) ===
    "conda",
    "conda_build",
    "setuptools",
    "pip",
    "wheel",
    "pkg_resources",
    "cryptography",
    "paramiko",
    "fabric",
    "requests_oauthlib",
    "oauthlib",
    "tornado",
    "spyder",
    "spyder_kernels",
    "aiobotocore",
    "botocore",
    "boto3",
    "s3fs",
    "gcsfs",
    "google",
    "azure",
    "cloudpickle",
    "lz4",
    "zstd",
    "blosc",
    "imageio",
    "imagecodecs",
    "tifffile",
    "skimage",
    "scikit-image",
    "cv2",
    "opencv",
    "nltk",
    "spacy",
    "tensorflow",
    "torch",
    "keras",
]


# ═══════════════════════════════════════════════════════════════════════════
# 📝 TEMPLATE CONTENT
# ═══════════════════════════════════════════════════════════════════════════

START_BAT_TEMPLATE = r"""@echo off
REM Check if Windows Terminal is available and use it for better emoji/color support
where wt >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Windows Terminal found - relaunch in it if not already in wt
    if "%WT_SESSION%"=="" (
        start "" wt new-tab -d . cmd /k call "%~nx0"
        exit /b
    )
)

title Zone Coverage Visualization Server
chcp 65001 >nul 2>&1

echo.
echo   ZONE COVERAGE VISUALIZATION
echo   ============================
echo.
echo   Starting server... please wait
echo   (First run may take 10-15 seconds to extract files)
echo.
echo   Browser will open automatically at http://127.0.0.1:5051
echo.
echo   To stop: Close this window or press Ctrl+C
echo.

REM Open browser after 5 second delay (gives server time to extract and start)
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:5051"

REM Run the server from the same directory as this batch file
"%~dp0zone_coverage_server.exe"

echo.
echo Server stopped.
pause
"""

README_TEMPLATE = """ZONE COVERAGE VISUALIZATION
===========================

QUICK START:
1. Double-click START.bat
2. Wait for browser to open (http://127.0.0.1:5051)
3. Use the map to view/edit borehole positions

TO STOP:
- Close the black command window, OR
- Press Ctrl+C in the server window

FEATURES:
- View zones and borehole positions on interactive map
- Drag boreholes to new locations
- See coverage update in real-time
- Toggle zone visibility with checkboxes
- Export modified positions to CSV
- Save positions for next session (persists across restarts)

SAVING YOUR CHANGES:
- Click "Export CSV" to download your borehole positions
- Place the exported CSV in the "Saved Positions" folder (next to START.bat)
- On next startup, the server will load positions from that CSV instead
- To reset to original data: remove the CSV from "Saved Positions"

TROUBLESHOOTING:
- If browser doesn't open: manually go to http://127.0.0.1:5051
- If "port in use" error: close the server window first, then try again
- If blocked by antivirus: add exception for zone_coverage_server.exe
- If SmartScreen warning: click "More info" then "Run anyway"
- If "Access Denied": right-click START.bat -> Run as administrator

DATA:
- Current data is in Data/zone_coverage_data.json
- To update data, replace this file and restart the server

Generated: {timestamp}
Version: 1.0
"""


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 BUILD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def check_pyinstaller() -> bool:
    """Check if PyInstaller is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"📦 PyInstaller version: {version}")
            return True
        return False
    except Exception:
        return False


def install_pyinstaller() -> bool:
    """Attempt to install PyInstaller via pip."""
    logger.info("📦 Installing PyInstaller...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("✅ PyInstaller installed successfully")
            return True
        else:
            logger.error(f"❌ pip install failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to install PyInstaller: {e}")
        return False


def build_pyinstaller_command(source_dir: Path, work_dir: Path) -> list:
    """
    Build the PyInstaller command with all necessary flags.

    Args:
        source_dir: Directory containing server.py (zone_coverage_viz/)
        work_dir: Parent directory to run PyInstaller from (Gap_Analysis_EC7/)

    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "zone_coverage_server",
        "--console",  # Show console for server output
        "--noconfirm",  # Overwrite without asking
    ]

    # Add data files (templates folder) - path relative to work_dir
    templates_dir = source_dir / "templates"
    if templates_dir.exists():
        # Windows uses ; as separator, Unix uses :
        separator = ";" if sys.platform == "win32" else ":"
        # Path from work_dir perspective
        cmd.extend(["--add-data", f"zone_coverage_viz/templates{separator}templates"])

    # CRITICAL: Bundle pyproj PROJ data (proj.db) - without this,
    # Transformer.from_crs() hangs or crashes in the frozen exe
    cmd.extend(["--collect-data", "pyproj"])

    # Add hidden imports
    for hidden in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", hidden])

    # Add exclusions
    for exclude in EXCLUDE_MODULES:
        cmd.extend(["--exclude-module", exclude])

    # Add the main script - path from work_dir perspective
    cmd.append("zone_coverage_viz/server.py")

    return cmd


def run_pyinstaller(source_dir: Path) -> Optional[Path]:
    """
    Run PyInstaller to create executable.

    Args:
        source_dir: Directory containing server.py (zone_coverage_viz/)

    Returns:
        Path to created executable, or None if failed
    """
    logger.info("🔨 Running PyInstaller (this may take 1-3 minutes)...")

    # Run from parent directory so 'from zone_coverage_viz.xxx' imports work
    work_dir = source_dir.parent
    cmd = build_pyinstaller_command(source_dir, work_dir)
    logger.info(f"📋 Command: {' '.join(cmd[:6])}...")

    try:
        # Run from parent dir (Gap_Analysis_EC7) so imports resolve correctly
        result = subprocess.run(cmd, cwd=work_dir, timeout=600)  # 10 minute timeout

        if result.returncode != 0:
            logger.error(f"❌ PyInstaller failed with code {result.returncode}")
            return None

        # Find the executable - now in work_dir/dist/
        exe_path = work_dir / "dist" / "zone_coverage_server.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Created executable: {exe_path} ({size_mb:.1f} MB)")
            return exe_path

        logger.error("❌ Executable not found after PyInstaller completed")
        return None

    except subprocess.TimeoutExpired:
        logger.error("❌ PyInstaller timed out after 10 minutes")
        return None
    except Exception as e:
        logger.error(f"❌ PyInstaller error: {e}")
        return None


def cleanup_pyinstaller_artifacts(source_dir: Path) -> None:
    """Remove PyInstaller build artifacts from source directory and parent."""
    logger.info("🧹 Cleaning up build artifacts...")

    work_dir = source_dir.parent
    for cleanup_name in ["build", "dist", "__pycache__"]:
        # Clean from both source and parent dir
        for base_dir in [source_dir, work_dir]:
            cleanup_path = base_dir / cleanup_name
            if cleanup_path.exists():
                shutil.rmtree(cleanup_path, ignore_errors=True)

    # Clean spec file from parent dir
    spec_file = work_dir / "zone_coverage_server.spec"
    if spec_file.exists():
        spec_file.unlink()

    spec_file = source_dir / "zone_coverage_server.spec"
    if spec_file.exists():
        spec_file.unlink()


def build_portable(
    output_dir: str = "ZoneCoverageViz_Portable",
    include_data: bool = True,
    clean_build: bool = True,
    auto_install_pyinstaller: bool = True,
    cleanup_artifacts: bool = True,
) -> Path:
    """
    Build a portable distribution of zone_coverage_viz.

    Args:
        output_dir: Name of output folder (created next to this script)
        include_data: Whether to copy Data/zone_coverage_data.json
        clean_build: Whether to delete existing output folder first
        auto_install_pyinstaller: Install PyInstaller if missing
        cleanup_artifacts: Remove build/dist folders after completion

    Returns:
        Path to the created portable folder

    Raises:
        RuntimeError: If build fails
    """
    # === Resolve paths ===
    script_dir = Path(__file__).parent.resolve()
    viz_dir = script_dir / "zone_coverage_viz"
    output_path = script_dir / output_dir

    logger.info(f"🚀 Building portable distribution")
    logger.info(f"   Source: {viz_dir}")
    logger.info(f"   Output: {output_path}")

    # === Validate source exists ===
    if not viz_dir.exists():
        raise RuntimeError(f"Source directory not found: {viz_dir}")

    if not (viz_dir / "server.py").exists():
        raise RuntimeError(f"server.py not found in {viz_dir}")

    # === Step 1: Check/Install PyInstaller ===
    if not check_pyinstaller():
        if auto_install_pyinstaller:
            if not install_pyinstaller():
                raise RuntimeError(
                    "Failed to install PyInstaller. "
                    "Try manually: pip install pyinstaller"
                )
        else:
            raise RuntimeError(
                "PyInstaller not installed. Run: pip install pyinstaller"
            )

    # === Step 2: Clean output directory ===
    if clean_build and output_path.exists():
        logger.info("🧹 Removing existing output directory...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # === Step 3: Run PyInstaller ===
    exe_path = run_pyinstaller(viz_dir)
    if exe_path is None:
        raise RuntimeError("PyInstaller build failed - see errors above")

    # === Step 4: Copy executable to output ===
    logger.info("📦 Copying executable to output folder...")
    shutil.copy2(exe_path, output_path / "zone_coverage_server.exe")

    # === Step 5: Copy data files ===
    if include_data:
        data_src = script_dir / "Output" / "zone_coverage_data.json"
        if data_src.exists():
            data_dst = output_path / "Data"
            data_dst.mkdir(exist_ok=True)
            shutil.copy2(data_src, data_dst / "zone_coverage_data.json")
            logger.info(
                f"📊 Copied data file ({data_src.stat().st_size / 1024:.1f} KB)"
            )
        else:
            logger.warning(f"⚠️ Data file not found: {data_src}")
            logger.warning(
                "   Run main.py first to generate data, or copy manually later"
            )

    # Create "Saved Positions" folder at portable root for CSV override
    saved_dir = output_path / "Saved Positions"
    saved_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\U0001f4c2 Created Saved Positions/ folder")

    # === Step 6: Create batch launcher ===
    (output_path / "START.bat").write_text(START_BAT_TEMPLATE, encoding="utf-8")
    logger.info("📝 Created START.bat")

    # === Step 7: Create README ===
    readme_content = README_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    (output_path / "README.txt").write_text(readme_content, encoding="utf-8")
    logger.info("📝 Created README.txt")

    # === Step 8: Cleanup build artifacts ===
    if cleanup_artifacts:
        cleanup_pyinstaller_artifacts(viz_dir)

    # === Done ===
    total_size = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ PORTABLE BUILD COMPLETE")
    logger.info(f"   Location: {output_path}")
    logger.info(f"   Total size: {total_size:.1f} MB")
    logger.info("")
    logger.info("   To test: Double-click START.bat")
    logger.info("   To distribute: Zip the entire folder")
    logger.info("=" * 60)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# 🚀 CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build portable zone_coverage_viz distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python portable_builder.py
    python portable_builder.py --output MyBuild
    python portable_builder.py --no-data --no-clean
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ZoneCoverageViz_Portable",
        help="Output directory name (default: ZoneCoverageViz_Portable)",
    )
    parser.add_argument(
        "--no-data", action="store_true", help="Don't include data files in output"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Don't remove existing output directory"
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep PyInstaller build/dist folders",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show debug output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    try:
        result = build_portable(
            output_dir=args.output,
            include_data=not args.no_data,
            clean_build=not args.no_clean,
            cleanup_artifacts=not args.keep_artifacts,
        )
        return 0
    except Exception as e:
        logger.error(f"\n❌ BUILD FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
