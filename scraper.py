import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import csv



def get_article_links(main_page_url):
    """
    Step 1: Crawls the main page to find all unique article URLs.
    """
    print(f"Crawling main page: {main_page_url}")
    try:
        response = requests.get(main_page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- IMPORTANT: You must find the correct selector for your site ---
        # Use your browser's "Inspect" tool to find the tag and class for the links.
        # This is just an example selector.
        link_elements = soup.find_all('a', class_='elementor-post__read-more')
        
        article_links = set() # Using a set to automatically handle duplicates
        for link_element in link_elements:
            href = link_element.get('href')
            if href:
                # Convert relative URL (e.g., /article/123) to an absolute URL
                absolute_url = urljoin(main_page_url, href)
                article_links.add(absolute_url)
                
        print(f"Found {len(article_links)} unique article links.")
        return list(article_links)
        
    except requests.RequestException as e:
        print(f"Error fetching main page: {e}")
        return []




    
def scrape_article_content(article_url):
    """
    Step 2: Scrapes the content from a single article URL.
    """
    print(f"  > Scraping article: {article_url}")
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- IMPORTANT: Find the correct selector for the article body ---
        # This is just an example selector.
        content_element = soup.find('div', class_='elementor-element elementor-element-47b8612 e-con-full e-flex e-con e-child')
        
        if content_element:
            # This is where your cleaning pipeline would go
            clean_text = ' '.join(content_element.get_text().split())
            return clean_text
        else:
            content_element = soup.find('div',class_='elementor-column elementor-col-33 elementor-top-column elementor-element elementor-element-45386381')
            if content_element:
            # This is where your cleaning pipeline would go
                clean_text = ' '.join(content_element.get_text().split())
                return clean_text
            else:
                print(f"  > Could not find article content at {article_url}")
                return None
            
    except requests.RequestException as e:
        print(f"  > Error scraping article {article_url}: {e}")
        return None


def get_main_url():
    MAIN_URL_LIST = ["https://imaa-institute.org/publications/"]

    for i in range(2,66):
        MAIN_URL_LIST.append(f"https://imaa-institute.org/publications/?e-page-8fbddee={i}")

    return MAIN_URL_LIST




# Define the path to your saved file
file_path = 'scraped_articles.json'
all_article_data = [] # Initialize as an empty list by default

try:
    # Open the file in read mode ('r')
    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load() reads the file and converts the JSON data
        # back into a Python list of dictionaries
        all_article_data = json.load(f)
    
    print(f"Successfully loaded {len(all_article_data)} articles from {file_path}")

except FileNotFoundError:
    print(f" Warning: File not found at {file_path}. Starting with an empty list.")
except json.JSONDecodeError:
    print(f" Error: Could not decode JSON from {file_path}. The file might be corrupted or empty.")

# Now the 'all_article_data' variable holds all the data you previously saved,
# and you can use it for your RAG pipeline.
# For example, you can access the first article's content:
# if all_article_data:
#     print(all_article_data[0]['content'])



# Define the path to your saved file
file_path = 'all_links.json'
all_link_data = [] # Initialize as an empty list by default

try:
    # Open the file in read mode ('r') with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load() reads the file and converts the JSON array
        # into a Python list of strings.
        all_link_data = json.load(f)
    
    print(f"Successfully loaded {len(all_link_data)} links from {file_path}")

except FileNotFoundError:
    print(f" Warning: The file '{file_path}' was not found. Starting with an empty list.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{file_path}'. The file might be empty or corrupted.")

# The 'all_link_data' variable now holds your list of URLs
# You can now use it like any other Python list.
if all_link_data:
    print("\nFirst link in the list is:")
    print(all_link_data[0])

    print("\nTotal links loaded:")
    print(len(all_link_data))




def scrape_data():
    # global all_article_data
    MAIN_URL_LIST = get_main_url();
    for main_url in MAIN_URL_LIST:

        MAIN_WEBSITE_URL = main_url
        
        # Step 1: Get all the links
        all_links = get_article_links(MAIN_WEBSITE_URL)
        
        # Step 2: Loop through links and scrape each one
        
        for link in all_links:

            if link in all_link_data:
                print("article already there, skiped")
                continue
            
            content = scrape_article_content(link)


            if content:
                all_link_data.append(link)
                print("new article, surfing...")
                all_article_data.append({'url': link, 'content': content})
            
            
            # Wait a second between requests.
            time.sleep(1) 
            
        print(f"\nSuccessfully scraped {len(all_article_data)} articles.")
    # Now `all_article_data` is a list of dictionaries ready to be saved.
    # For example: print(all_article_data[0])



def save_to_json(data_list, filename):
    """Saves a list of dictionaries to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # json.dump writes the data to the file
            # indent=4 makes the file nicely formatted and easy to read
            json.dump(data_list, f, indent=4)
        print(f" Data successfully saved to {filename}")
    except Exception as e:
        print(f" Error saving to JSON file: {e}")

# --- How to use it in your main() function ---
# Place this at the very end of your main() function:

# if all_article_data:
#     save_to_json(all_article_data)


def save_to_csv(data_list, filename):
    """Saves a list of dictionaries to a CSV file."""
    if not data_list:
        print("No data to save.")
        return

    try:
        # The keys of the first dictionary are used as the CSV headers
        headers = data_list[0].keys()
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            # DictWriter maps dictionaries to CSV rows
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()  # Writes the column titles (e.g., 'url', 'content')
            writer.writerows(data_list) # Writes all the data rows
            
        print(f" Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV file: {e}")

# --- How to use it in your main() function ---
# Place this at the very end of your main() function:

# if all_article_data:
#     save_to_csv(all_article_data)

