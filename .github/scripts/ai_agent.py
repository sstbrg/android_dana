import os
import sys
import requests
import re
from pathlib import Path
import google.auth
import google.auth.transport.requests

# --- Configuration ---
# Get the issue details from environment variables set by the GitHub Action
ISSUE_TITLE = os.getenv("ISSUE_TITLE")
ISSUE_BODY = os.getenv("ISSUE_BODY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") # Get GCP Project ID from environment
LLM_API_URL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.5-flash:generateContent"
FILE_SEPARATOR = "---FILE_SEPARATOR---"

def get_gcp_auth_session():
    """
    Authenticates with GCP using Application Default Credentials (ADC)
    and returns an authorized session object.
    """
    try:
        # This will automatically find the credentials provided by
        # the google-github-actions/auth action in the CI/CD environment.
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
        return authed_session
    except google.auth.exceptions.DefaultCredentialsError:
        print("Error: Could not find Google Cloud credentials.")
        print("Please ensure you are running in a configured GCP environment or have set up Application Default Credentials.")
        sys.exit(1)

def call_llm(prompt: str, session: requests.Session) -> str:
    """
    Calls the configured LLM API with a given prompt using an authenticated session.

    Args:
        prompt: The prompt to send to the language model.
        session: The authenticated requests.Session object.

    Returns:
        The text content from the LLM's response.
    """
    print("Sending prompt to LLM...")
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = session.post(LLM_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        
        response_data = response.json()
        
        candidates = response_data.get('candidates', [])
        if not candidates:
            raise ValueError("No candidates found in LLM response.")
            
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        if not parts:
            raise ValueError("No parts found in LLM response content.")

        print("LLM response received successfully.")
        return parts[0].get('text', '')

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        if e.response is not None:
            print(f"Response Body: {e.response.text}")
        sys.exit(1)
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Full Response: {response_data}")
        sys.exit(1)

def parse_and_write_files(response_text: str):
    """
    Parses the LLM response text, which contains multiple files,
    and writes each file to its specified path.

    Args:
        response_text: The raw text from the LLM containing file blocks.
    """
    print("Parsing response and writing files...")
    file_blocks = response_text.split(FILE_SEPARATOR)

    for block in file_blocks:
        if not block.strip():
            continue

        # Use regex to find the file path comment (for both # and <!-- --> style comments)
        match = re.search(r"^\s*(?:#|<!--)\s*FILE_PATH:\s*(.*?)\s*(?:-->)?", block, re.MULTILINE)
        if not match:
            print(f"Warning: Could not find FILE_PATH in block:\n---\n{block[:200]}...\n---")
            continue

        file_path_str = match.group(1).strip()
        
        content_start_index = match.end()
        clean_block = block[content_start_index:].strip()
        if clean_block.startswith("```"):
            first_line_end = clean_block.find('\n') + 1
            clean_block = clean_block[first_line_end:]
        if clean_block.endswith("```"):
            clean_block = clean_block[:-3]
        
        content = clean_block.strip()

        try:
            file_path = Path(file_path_str)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            print(f"Successfully wrote to {file_path}")

        except Exception as e:
            print(f"Error writing file {file_path_str}: {e}")
            sys.exit(1)

def main():
    """
    Main function to orchestrate the AI agent's tasks.
    """
    if not ISSUE_TITLE or not ISSUE_BODY or not GCP_PROJECT_ID:
        print("Error: ISSUE_TITLE, ISSUE_BODY, or GCP_PROJECT_ID environment variables not set.")
        sys.exit(1)

    # Get an authenticated session for GCP
    authed_session = get_gcp_auth_session()

    # --- Step 1: Generate Application Code ---
    print("\n--- Step 1: Generating Application Code ---")
    code_gen_prompt = f"""
    You are an expert Android developer specializing in Kotlin and modern Android practices.
    Your task is to generate all necessary files for a new feature based on a GitHub issue.

    GitHub Issue Title: '{ISSUE_TITLE}'
    GitHub Issue Body:
    ---
    {ISSUE_BODY}
    ---

    Instructions:
    1.  Generate complete, production-ready Kotlin and XML layout files.
    2.  Ensure the code is clean, well-commented, and follows standard Android architecture patterns (e.g., MVVM if applicable).
    3.  IMPORTANT: Your response MUST be a single block of text. Separate each file's content with the exact delimiter: '{FILE_SEPARATOR}'.
    4.  At the beginning of each file's content, you MUST include a comment indicating its full, relative path from the project root. The format is critical. Example:
        # FILE_PATH: app/src/main/java/com/example/myapp/MyNewActivity.kt
    5.  For XML files, use a comment like this:
        <!-- FILE_PATH: app/src/main/res/layout/activity_my_new.xml -->
    """
    
    generated_code_response = call_llm(code_gen_prompt, authed_session)
    parse_and_write_files(generated_code_response)
    print("Application code generation complete.")

    # --- Step 2: Generate Unit Tests ---
    print("\n--- Step 2: Generating Unit Tests ---")
    test_gen_prompt = f"""
    You are an expert Android test engineer. Your task is to write comprehensive unit tests for the provided application code.

    Application Code to Test:
    ---
    {generated_code_response}
    ---

    Instructions:
    1.  Use JUnit 5 and Mockito for testing.
    2.  Generate a complete test file that covers the logic in the provided code. Include tests for happy paths and edge cases.
    3.  As before, your response must be a single block of text, and you MUST start the file content with a file path comment. Example:
        # FILE_PATH: app/src/test/java/com/example/myapp/MyNewActivityTest.kt
    """
    
    generated_test_response = call_llm(test_gen_prompt, authed_session)
    parse_and_write_files(generated_test_response)
    print("Unit test generation complete.")
    print("\nAI Agent finished successfully.")

if __name__ == "__main__":
    main()
```

### **Setup Guide for GCP and GitHub Actions**

Follow these steps to configure the environment for the updated script.

#### **Step 1: Configure Your GCP Project**

1.  **Enable APIs:** In your GCP project, make sure the **Vertex AI API** is enabled. You can do this from the GCP Console by navigating to "APIs & Services" > "Library" and searching for it.
2.  **Create a Service Account:**
    * Go to "IAM & Admin" > "Service Accounts".
    * Click **+ CREATE SERVICE ACCOUNT**.
    * Give it a name (e.g., `github-actions-agent`) and a description.
    * Click **CREATE AND CONTINUE**.
    * Grant it the **Vertex AI User** role. This gives it permission to call the Gemini model.
    * Click **DONE**. You do *not* need to create a key file.

3.  **Set up Workload Identity Federation:** This is the secure, keyless way to connect GitHub Actions to GCP.
    * Go to "IAM & Admin" > "Workload Identity Federation".
    * Click **CREATE POOL**, give it a name (e.g., `github-pool`), and continue.
    * Under "Add a provider to a pool", select **OIDC**.
    * **Provider Name:** `github-provider`
    * **Issuer (URL):** `https://token.actions.githubusercontent.com`
    * **Audience:** Leave as default.
    * Under "Attribute mapping", add the following mappings:
        * `google.subject` -> `assertion.sub`
        * `attribute.actor` -> `assertion.actor`
        * `attribute.repository` -> `assertion.repository`
    * Save the provider.
    * Now, go back to the Service Account you created in step 2.
    * Go to the **Permissions** tab.
    * Click **+ GRANT ACCESS**.
    * For "New principals", paste the following, replacing `[YOUR_GITHUB_ORG/USER]/[YOUR_REPO]` with your details:
        ```
        principalSet://iam.googleapis.com/projects/[PROJECT_NUMBER]/locations/global/workloadIdentityPools/github-pool/subject/repo:[YOUR_GITHUB_ORG/USER]/[YOUR_REPO]:ref:refs/heads/main
        ```
    * Grant it the **Workload Identity User** role. This allows GitHub Actions from your repo to impersonate this service account.

#### **Step 2: Configure GitHub Secrets**

In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add the following secrets:

* `GCP_PROJECT_ID`: Your Google Cloud Project ID.
* `GCP_WORKLOAD_IDENTITY_PROVIDER`: The full name of the Workload Identity Provider you created. It will look like `projects/[PROJECT_NUMBER]/locations/global/workloadIdentityPools/github-pool/providers/github-provider`.
* `GCP_SERVICE_ACCOUNT_EMAIL`: The email address of the service account you created (e.g., `github-actions-agent@[PROJECT_ID].iam.gserviceaccount.com`).

#### **Step 3: Update Your GitHub Actions Workflow**

Finally, modify your `.github/workflows/android-ai-ci.yml` file to use the new GCP authentication method.

```yaml
name: Autonomous Android AI Engineer (GCP)

on:
  issues:
    types: [labeled]

jobs:
  build-feature:
    if: github.event.label.name == 'autogen-feature'
    runs-on: ubuntu-latest
    permissions:
      contents: write      # To push code and create PRs
      pull-requests: write # To create PRs
      id-token: write      # To authenticate with GCP

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT_EMAIL }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Python dependencies
        run: pip install requests google-auth

      - name: Run AI Agent to Generate Code and Tests
        env:
          GCP_PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
          ISSUE_TITLE: ${{ github.event.issue.title }}
          ISSUE_BODY: ${{ github.event.issue.body }}
        run: python .github/scripts/ai_agent.py

      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
          cache: gradle

      - name: Grant execute permission for gradlew
        run: chmod +x ./gradlew

      - name: Build, Test, and Run Static Analysis
        run: ./gradlew check test

      - name: Upload SpotBugs Report on Failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: spotbugs-report
          path: app/build/reports/spotbugs/

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: "feat: AI generated feature for #${{ github.event.issue.number }}"
          title: "AI Feature: ${{ github.event.issue.title }}"
          body: |
            This PR was automatically generated by a GCP-powered AI agent in response to issue #${{ github.event.issue.number }}.

            **Validation Checks:**
            - ✅ Build Successful
            - ✅ Unit Tests Passed
            - ✅ SpotBugs Static Analysis Passed

            Please review the generated code for logic, style, and correctness.
          branch: "feature/ai-issue-${{ github.event.issue.number }}"
          base: "main"
          labels: "ai-generated"

```

With these changes, your agent is now securely authenticated with GCP and ready to run in a professional cloud environme
