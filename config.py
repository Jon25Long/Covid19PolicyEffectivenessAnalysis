# Project Configuration
# This file defines paths and settings for the project
# DO NOT hardcode paths in notebooks - always use these variables

# Project Information
project_name = "covid19-tracker"
project_version = "1.0.0"

# Data Information
# COVID-19 data from Our World in Data (OWID)
# Data currency: 2020-01-01 through 2024-08-04
# Source: https://github.com/owid/covid-19-data

# Directory Structure (relative to project root)
notebooks_dir = "notebooks"
scripts_dir = "scripts"
data_dir = "Data"
output_dir = "output"
docs_dir = "docs"
images_dir = "Images"

# Paths Module (for import)
class Paths:
    """Project paths for easy import in notebooks."""
    NOTEBOOKS = notebooks_dir
    SCRIPTS = scripts_dir
    DATA = data_dir
    OUTPUT = output_dir
    DOCS = docs_dir
    IMAGES = images_dir
    
    # External tools (relative to project root)
    TOOLS = "../../tools-and-frameworks"
    NAVIGATOR = "../../tools-and-frameworks/navigator"
    
    # Data files
    ISO_CODES = f"{data_dir}/iso_country_codes.csv"
    GEO_DATA = f"{data_dir}/world_latitude_longitude.csv"

# Generation Settings
use_semantic_matching = True
country_code_threshold = 0.85
create_visualizations = True
create_geographic_maps = True

# Geoapify API Key (add your key here)
geoapify_key = "ADD API KEY HERE"
