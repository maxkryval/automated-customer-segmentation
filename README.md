
# Automated Customer Segmentation

This repository contains a bachelor's thesis demo pipeline for automated customer segmentation from natural language campaign descriptions.

## Setup and run
1. Make sure to have python ```3.12+``` installed and ensure its alias its PATH alias is ```python3```. Unless it's not ```python3```, adapt ```run.sh``` file to a relevant alias of python.
2. Clone the repository.
3. Fill ```OPENAI_API_KEY``` in run_pipeline.py file.
4. Fill ```CAMPAIGN_PROMPT``` string variable with the desired campaign to test in ```run_pipeline.py``` file.
5. Optionally fill ````CAMPAIGN_ID``` field to identify the campaign in the output folder after the results come through.
6. In terminal - ```bash run.sh```
