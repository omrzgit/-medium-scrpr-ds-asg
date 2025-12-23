import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse
import pandas as pd
import time
import os

def fetch_url(url, max_retries=3):
    """
    Fetches the content of the URL with a retry mechanism and robust headers.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'Connection': 'keep-alive',
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                delay = 5 * (attempt + 1) # 5s, 10s, 15s
                print(f"  -> Attempt {attempt + 1}: 403 Forbidden. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Error fetching {url}: {e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    return None

def extract_medium_data(url):
    """
    Scrapes a single Medium article URL to extract required data points.
    """
    response = fetch_url(url)
    if not response:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    article_data = {'URL': url}

    # --- 1. Extract data from the embedded JSON in the HTML source (Most Reliable) ---
    data_script = soup.find('script', {'id': '__NEXT_DATA__'})
    post_data = None
    
    if data_script:
        try:
            json_data = json.loads(data_script.string)
            
            # Navigate the JSON structure to find the article data
            page_props = json_data.get('props', {}).get('pageProps', {})
            
            # Try to find the post data in common locations
            post_data = page_props.get('pageData', {}).get('post') or page_props.get('post')
            
            if post_data:
                article_data['Title'] = post_data.get('title')
                article_data['Subtitle'] = post_data.get('virtuals', {}).get('subtitle')
                article_data['Claps'] = post_data.get('virtuals', {}).get('totalClapCount')
                article_data['Reading Time'] = post_data.get('virtuals', {}).get('readingTime')
                article_data['Keywords'] = [tag['name'] for tag in post_data.get('virtuals', {}).get('tags', [])]
                
                # Author Info
                creator_id = post_data.get('creatorId')
                user = page_props.get('pageData', {}).get('user')
                if user and user.get('userId') == creator_id:
                    article_data['Author Name'] = user.get('name')
                    article_data['Author URL'] = f"https://medium.com/@{user.get('username')}"
                else:
                    # Fallback to find author in post data
                    author = post_data.get('creator')
                    if author:
                        article_data['Author Name'] = author.get('name')
                        article_data['Author URL'] = f"https://medium.com/@{author.get('username')}"
                
                # Text extraction from JSON (list of paragraphs)
                text_paragraphs = post_data.get('content', {}).get('bodyModel', {}).get('paragraphs', [])
                article_data['Text'] = " ".join([p.get('text', '') for p in text_paragraphs])
            
        except json.JSONDecodeError:
            print(f"Could not decode JSON from script tag for {url}")
        except Exception as e:
            print(f"Error processing JSON data for {url}: {e}")

    # --- 2. Fallback to HTML parsing for data not found in JSON ---
    
    # Title fallback
    if not article_data.get('Title'):
        title_tag = soup.find('h1')
        article_data['Title'] = title_tag.text.strip() if title_tag else 'N/A'
    
    # Subtitle fallback
    if not article_data.get('Subtitle'):
        subtitle_tag = soup.find('h2')
        article_data['Subtitle'] = subtitle_tag.text.strip() if subtitle_tag else 'N/A'

    # Text fallback (if JSON extraction failed)
    if not article_data.get('Text'):
        main_content = soup.find('div', role='main') or soup.find('article')
        if main_content:
            article_data['Text'] = main_content.get_text(separator=' ', strip=True)
        else:
            article_data['Text'] = soup.body.get_text(separator=' ', strip=True)[:500] + '...' # First 500 chars as a fallback

    # --- 3. Image and Link Counting (HTML Parsing) ---
    
    # Count images and extract URLs
    image_tags = soup.find_all('img')
    # Filter out small/logo images
    article_images = [img['src'] for img in image_tags if img.get('src') and ('cdn-images-1.medium.com' in img['src'] or 'miro.medium.com' in img['src']) and 'w=40' not in img['src']]
    article_data['No. of images'] = len(article_images)
    article_data['Image URLs'] = ", ".join(article_images)

    # Count external links
    article_domain = urlparse(url).netloc
    external_links = 0
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        link_domain = urlparse(href).netloc
        # Check if the link has a domain and it's not the article's domain (or a subdomain of medium.com)
        if link_domain and not (link_domain == article_domain or link_domain.endswith('.medium.com') or link_domain.endswith('towardsdatascience.com')):
            external_links += 1
    article_data['No. of external links'] = external_links

    # --- 4. Final Formatting and Cleanup ---
    
    final_result = {
        'URL': url,
        'Title': article_data.get('Title', 'N/A'),
        'Subtitle': article_data.get('Subtitle', 'N/A'),
        'Text': article_data.get('Text', 'N/A'),
        'No. of images': article_data.get('No. of images', 0),
        'Image URLs': article_data.get('Image URLs', 'N/A'),
        'No. of external links': article_data.get('No. of external links', 0),
        'Author Name': article_data.get('Author Name', 'N/A'),
        'Author URL': article_data.get('Author URL', 'N/A'),
        'Claps': article_data.get('Claps', 0),
        'Reading Time': f"{int(article_data.get('Reading Time', 0))} min" if article_data.get('Reading Time') else 'N/A',
        'Keywords': ", ".join(article_data.get('Keywords', []))
    }
    
    return final_result

def scrape_urls_to_csv(urls_file="urls.txt", output_filename="scrapping_results.csv"):
    """
    Main function to scrape a list of URLs from a file and save the results to a CSV file.
    """
    if not os.path.exists(urls_file):
        print(f"Error: URL file '{urls_file}' not found.")
        return

    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"Error: URL file '{urls_file}' is empty.")
        return

    results = []
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Scraping: {url}")
        data = extract_medium_data(url)
        if data:
            results.append(data)
            print(f"  -> Success: {data['Title']}")
        else:
            print(f"  -> Failed to scrape: {url}")
        
        # Be polite and wait a significant time between requests
        time.sleep(10)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully scraped {len(results)} articles and saved to {output_filename}")
    else:
        print("\nNo articles were successfully scraped.")

if __name__ == '__main__':
    scrape_urls_to_csv()
