import re
import fitz  # PyMuPDF for PDF handling
import docx
import os
import requests
import base64
import json

# Network hardening: every external call gets a timeout so a hung API can
# never hang (and crash) the /api/resume/upload endpoint. One bad repo must
# not kill the whole batch either — failures are isolated per repo.
HTTP_TIMEOUT = float(os.getenv("SCRAP_TIMEOUT", "10"))
MAX_REPOS = int(os.getenv("SCRAP_MAX_REPOS", "10"))


def _github_headers():
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repository_details(repo_url):
    """Fetch repository details like description and README content.

    Never raises: returns [] when the repo is unreachable/private/invalid.
    Returns: [description (str), readme (str), language (str)]
    """
    try:
        parts = repo_url.strip('/').split('/')
        if len(parts) != 5:
            print(f"Invalid repository URL format: {repo_url}")
            return []

        owner, repo = parts[3], parts[4]
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url, headers=_github_headers(), timeout=HTTP_TIMEOUT)

        if response.status_code != 200:
            print(f"Error: Unable to fetch repository details for {repo_url} (HTTP {response.status_code})")
            return []

        repo_details = []
        data = response.json()
        project_data = data.get('description', 'No description available')
        repo_details.append(project_data)

        readme_url = f"{api_url}/readme"
        readme_response = requests.get(readme_url, headers=_github_headers(), timeout=HTTP_TIMEOUT)

        if readme_response.status_code == 200:
            try:
                readme_data = readme_response.json()
                decoded_content = base64.b64decode(readme_data['content']).decode('utf-8')
                repo_details.append(decoded_content)
            except Exception as e:
                repo_details.append(f"Error: Unable to decode README content ({e}).")
        else:
            repo_details.append("Error: Unable to fetch README content.")

        # Primary language powers RAG technology chunks; "" when unknown (appended last: index 2)
        repo_details.append(data.get('language') or "")
        return repo_details
    except requests.exceptions.RequestException as e:
        print(f"Error: Network failure fetching {repo_url} ({type(e).__name__}: {e})")
        return []
    except Exception as e:
        print(f"Error: Unexpected failure fetching {repo_url} ({e})")
        return []


def leetcode_details(username):
    """Fetch LeetCode problem-solving stats."""
    url = "https://leetcode.com/graphql/"
    headers = {"Content-Type": "application/json"}
    query = {
        "query": """
        query skillStats($username: String!) {
  matchedUser(username: $username) {
    tagProblemCounts {
      advanced {
        tagName
        tagSlug
        problemsSolved
      }
      intermediate {
        tagName
        tagSlug
        problemsSolved
      }
      fundamental {
        tagName
        tagSlug
        problemsSolved
      }
    }
  }
}
        """,
        "variables": {"username": username}
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(query), timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data.get("data", {}).get("matchedUser", {}), indent=4)
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def extract_links_from_pdf(pdf_path):
    """Extract all clickable and text-based links from a PDF file."""
    links = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            for link in page.get_links():
                if 'uri' in link and link['uri'].startswith("http"):
                    links.append(link['uri'])
            links.extend(extract_links_from_text(page.get_text("text")))
    except Exception as e:
        print(f"Error extracting PDF links: {e}")
    return list(set(links))


def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""


def extract_resume_text(file_path):
    """Return the full text of a PDF/DOCX resume (used by RAG ingestion).

    Link extractors only keep URLs; ingestion needs the body text.
    Returns "" when the file is missing or unreadable (never raises).
    """
    if not os.path.exists(file_path):
        return ""
    try:
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".pdf":
            doc = fitz.open(file_path)
            return "\n".join(page.get_text("text") for page in doc)
        return extract_text_from_docx(file_path)
    except Exception as e:
        print(f"Error extracting resume text: {e}")
        return ""


def extract_links_from_text(text):
    """Extract all links from text using regex."""
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    return url_pattern.findall(text)


def get_github_entries(file_path):
    """Like get_github_details but keeps the source URL with each result.

    Returns [{"url": link, "details": [description, readme, language]}].
    RAG ingestion needs the URL for repository metadata; the plain
    get_github_details() drops it.
    """
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return []

    _, ext = os.path.splitext(file_path)
    links = extract_links_from_pdf(file_path) if ext.lower() == ".pdf" else extract_links_from_text(extract_text_from_docx(file_path))
    github_links = [link for link in links if "github.com" in link and link.count('/') >= 4]
    if len(github_links) > MAX_REPOS:
        print(f"Found {len(github_links)} repos, fetching first {MAX_REPOS} (SCRAP_MAX_REPOS).")
        github_links = github_links[:MAX_REPOS]
    # Per-link isolation: repository_details never raises, but guard anyway.
    entries = []
    for link in github_links:
        try:
            entries.append({"url": link, "details": repository_details(link)})
        except Exception as e:
            print(f"Error: Skipping {link} ({e})")
            entries.append({"url": link, "details": []})
    return entries


def get_github_details(file_path):
    """Extract and return GitHub repository details from a resume."""
    return [e["details"] for e in get_github_entries(file_path)]


def get_leetcode_details(file_path):
    """Extract and return LeetCode user details from a resume."""
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return ""

    _, ext = os.path.splitext(file_path)
    links = extract_links_from_pdf(file_path) if ext.lower() == ".pdf" else extract_links_from_text(extract_text_from_docx(file_path))
    leetcode_links = [link for link in links if "leetcode.com" in link]
    if leetcode_links:
        raw = leetcode_links[0].rstrip("/").split("?")[0].split("#")[0]
        # Supports both https://leetcode.com/u/username and https://leetcode.com/username
        match = re.search(r'leetcode\.com/(?:u/)?([a-zA-Z0-9_-]+)/?$', raw)
        # Guard against problem-list URLs like leetcode.com/problems/...
        if match and match.group(1).lower() not in ("problems", "contest", "discuss", "explore", "u"):
            return leetcode_details(match.group(1)) if match else "Invalid URL"
        return "Invalid URL"
    return "No LeetCode link found."
