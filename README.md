Challenge 1b: Multi-Collection PDF Analysis
Overview
This project is an advanced PDF analysis solution that processes multiple, self-contained document collections. It extracts and prioritizes relevant content based on a specific persona and use case defined for each collection, adhering to all hackathon constraints including offline execution and model size limits.

Project Structure
The project is organized by collections, where each collection folder contains its own documents, input query, and will store its own output.

/Challenge_1b/
├── Collection 1/
│   ├── PDFs/                       # PDFs for this collection
│   └── challenge1b_input.json      # Input configuration for this collection
├── Collection 2/
│   ├── PDFs/
│   └── challenge1b_input.json
├── models/                         # AI models (downloaded once)
├── .dockerignore
├── Dockerfile
├── main.py
├── requirements.txt
└── setup_project.py

How to Run
1. Initial Setup (Online)
First, prepare the project environment. This step requires an internet connection.

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Run the setup script to install packages and download models
python setup_project.py

2. Prepare a Collection
For each collection folder (e.g., Collection 1), you must prepare two things:

a) Place PDFs in the PDFs folder:

Add all the relevant PDF documents for the collection into its designated PDFs subfolder (e.g., Collection 1/PDFs/).

b) Configure the challenge1b_input.json file:

Create or edit the challenge1b_input.json file inside the collection folder. It must follow this exact structure:

{
  "persona": {
    "role": "HR professional"
  },
  "job_to_be_done": {
    "task": "Create and manage fillable forms for onboarding and compliance."
  }
}

3. Running with Docker
This is the primary method for running the analysis. It ensures a consistent and isolated environment.

Build the Image (One-Time):
First, build the Docker image and name it adobe1b. This command only needs to be run once, or whenever you change the project's code or dependencies.

docker build -t adobe1b .

Run the Container:
This command runs the analysis on all available collections. It works by mounting your entire project directory into the container, which allows the script to access your collection folders and save the results back to your local machine.

docker run --rm -v ${pwd}:/app adobe1b

The output file, challenge1b_output.json, will be saved directly inside each respective collection folder after the container finishes its execution.