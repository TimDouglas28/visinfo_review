"""
#################################################
Module to scrape news content from Politifact.com
#################################################
"""

import time
import random
import pickle
import os
import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchWindowException

dummy=0

class PolitifactScraper:
    """
    A class to scrape news items from Politifact.com.
    Currently setup to scrape 'True' and 'Pants on Fire' 
    headlines but can be easily adapted to include others.

    Attributes:
        num_headlines (int): The total number of headlines to scrape.
        headlines_per_page (int): Number of headlines per page to scrape.
        true_url (str): URL for 'True' headlines.
        pants_fire_url (str): URL for 'Pants on Fire' headlines.
        driver (webdriver.Chrome): Selenium WebDriver instance.
    """

    def __init__(self, num_headlines, headlines_per_page=30):
        """
        Initializes the scraper.

        Args:
            num_headlines (int): The total number of headlines to scrape. 
                                Adjust this to collect more or fewer headlines.
            headlines_per_page (int): The number of headlines to scrape per page (default 30). 
        """
        self.num_headlines = num_headlines
        self.headlines_per_page = headlines_per_page
        self.true_url = "https://www.politifact.com/factchecks/list/?ruling=true"
        self.pants_fire_url = "https://www.politifact.com/factchecks/list/?ruling=pants-fire"
        self.driver = webdriver.Chrome()

    def _safe_get(self, url):
        """
        Safely open a webpage with timeout handling.

        Args:
            url (str): The URL to visit.
        """
        try:
            self.driver.set_page_load_timeout(15)
            self.driver.get(url)
        except Exception as e:
            print(f"Error loading page: {e}")
            self.driver.set_page_load_timeout(30)

    def _safe_click(self, element):
        """
        Safely click a web element with timeout handling.

        Args:
            element: Selenium WebElement to click.
        """
        try:
            self.driver.set_page_load_timeout(15)
            element.click()
        except Exception as e:
            print(f"Error clicking element: {e}")
            self.driver.set_page_load_timeout(30)

    def _extract_headlines_from_page(self, label):
        """
        Extract headlines from the current page based on the provided label.

        Args:
            label (str): The label for the headlines, e.g., 'True' or 'False'.

        Returns:
            list: A list of dictionaries containing headline data.
        """
        headlines_elements = self.driver.find_elements_by_css_selector(".m-statement")
        return [{
            "headline_text": headline.find_element_by_css_selector(".m-statement__quote a").text,
            "headline_link": headline.find_element_by_css_selector(".m-statement__quote a").get_attribute("href"),
            "headline_label": label,
            "source": headline.find_element_by_css_selector(".m-statement__name").text,
            "more": headline.find_element_by_css_selector(".m-statement__desc").text
        } for headline in headlines_elements]

    def _get_next_page(self):
        """
        Attempt to navigate to the next results page.

        Returns:
            bool: True if navigation occurred, False otherwise.
        """
        try:
            buttons = self.driver.find_elements_by_css_selector('a.c-button.c-button--hollow')

            for button in buttons:
                if button.text.strip().lower() == "next":
                    self.driver.execute_script("arguments[0].scrollIntoView();", button)
                    time.sleep(random.uniform(2, 5))  # Random delay to avoid detection
                    self.driver.execute_script("arguments[0].click();", button)
                    return True
            
            print("Next button not found.")
            return False

        except TimeoutException:
            print("Timeout error: Retrying after a delay...")
            time.sleep(10)  # Longer delay to prevent further timeouts
            return self._get_next_page()

        except NoSuchWindowException:
            print("Browser window closed unexpectedly. Restarting session...")
            self._restart_driver()  # Custom method to restart Selenium (see below)
            return False

        except Exception as e:
            print(f"Error clicking element: {e}")
            return False

    def _scrape_page(self, url, label):
        """
        Scrape pages starting from a URL until the required number of headlines is reached.


        Args:
            url (str): The URL of the page to scrape.
            label (str): The label for the headlines, e.g., 'True' or 'False'.

        Returns:
            list: A list of dictionaries containing headline data.
        """
        self._safe_get(url)
        all_headlines = []
        page_count = 0

        while len(all_headlines) < self.num_headlines:
            if page_count >= 100:  # Limit to 5 pages
                break

            page_headlines = self._extract_headlines_from_page(label)
            all_headlines.extend(page_headlines)

            if len(all_headlines) < self.num_headlines and self._get_next_page():
                page_count += 1
                time.sleep(2)  # Wait for the next page to load
            else:
                break

        return all_headlines[:self.num_headlines]

    def scrape(self):
        """
        Scrape both 'True' and 'Pants on Fire' labeled headlines.

        Returns:
            list: Shuffled list of scraped headlines.
        """
        true_headlines = []
        pants_fire_headlines = []

        while len(true_headlines) < self.num_headlines // 2:
            true_headlines.extend(self._scrape_page(self.true_url, label='True'))

        while len(pants_fire_headlines) < self.num_headlines // 2:
            pants_fire_headlines.extend(self._scrape_page(self.pants_fire_url, label='False'))

        all_headlines = true_headlines + pants_fire_headlines
        random.shuffle(all_headlines)

        return all_headlines

    def save_results(self, file_path='data/results.pickle'):
        """
        Save scraped headlines to a pickle file.

        Args:
            file_path (str): Path to the pickle file where results will be saved.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        all_headlines = self.scrape()
        with open(file_path, 'wb') as file:
            pickle.dump(all_headlines, file)
        print(f"Scraped headlines saved to {file_path}")

    def close(self):
        """Close the browser."""
        self.driver.quit()

class ImageExtractor:
    """
    A class to extract images from URLs in data.

    Attributes:
        driver (webdriver.Chrome): Selenium WebDriver instance.
    """

    def __init__(self):
        """Initializes the ImageExtractor with a WebDriver instance."""
        self.driver = webdriver.Chrome()

    def _get_image_and_text_from_page(self, url, num_paragraphs):
        """
        Extract the highest resolution image and first k paragraphs from the article.

        If no image is found, this function will return `None` for the image. The text extraction
        will proceed regardless of the image availability.

        Args:
            url (str): URL of the article.
            num_paragraphs (int): Number of paragraphs to extract.

        Returns:
            tuple: (image_url, article_text) where image_url is the URL of the image, 
                and article_text is a list of up to `num_paragraphs` paragraphs.
        """
        try:
            self.driver.get(url)

            # Locate the highest resolution image
            high_res_image = self.driver.find_element(By.CSS_SELECTOR, 'img.c-image__original.lozad')
            img_url = high_res_image.get_attribute("data-src") or high_res_image.get_attribute("src")

            # Extract the first three paragraphs from the article
            paragraphs = self.driver.find_elements(By.CSS_SELECTOR, 'article.m-textblock p')
            article_text = [p.text for p in paragraphs[:num_paragraphs]]  # Get first 3 paragraphs

            # Extract tags from webpage
            tag_elements = self.driver.find_elements(By.CSS_SELECTOR, 'ul.m-list.m-list--horizontal li.m-list__item span')
            tags = [tag.text for tag in tag_elements]

            return img_url, article_text, tags

        except Exception as e:
            print(f"Error extracting data from {url}: {e}")
            return None, None, None

    def extract_images_and_text(self, headlines, num_paragraphs):
        """
        Extracts images and first k paragraphs for each headline.

        Args:
            headlines (list): List of headlines with URLs.

        Returns:
            list: Updated headlines list with added 'image_url' and 'article_text'.
        """
        for headline in headlines:
            img_url, article_text, tags = self._get_image_and_text_from_page(headline['headline_link'], num_paragraphs)
            headline['image_url'] = img_url
            headline['article_text'] = article_text  # Save extracted text
            headline['tags'] = tags # save extracted article tags

        return headlines

    def close(self):
        """Close the browser."""
        self.driver.quit()

def download_images(data, folder='./imgs'):
    """
    Downloads images from the 'image_url' key in data and saves them in the specified folder.

    Args:
        data (list of dict): List of dictionaries containing 'image_url' and 'headline_label'.
        folder (str): Folder where images will be saved.

    Returns:
        list: List of tuples containing image names and indices for successfully downloaded images.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    true_counter = 1
    false_counter = 1
    downloaded_images = []

    for idx, entry in enumerate(data):
        image_url = entry.get('image_url')
        if not image_url:
            continue

        if entry.get('headline_label') in ['True', 'False']:  # Ensure label is valid
            is_true = entry['headline_label'] == 'True'
            image_name = f"t{true_counter:03}.jpg" if is_true else f"f{false_counter:03}.jpg"
            if is_true:
                true_counter += 1
            else:
                false_counter += 1

            image_path = os.path.join(folder, image_name)

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {image_name}")
                downloaded_images.append((image_name, idx))
            except requests.RequestException as e:
                print(f"Failed to download {image_url}: {e}")

    return downloaded_images

def save_data_as_json(data, downloaded_images, json_filename='./data/news.json'):
    """
    Saves the given data as a JSON file using only successfully downloaded images.

    Args:
        data (list of dict): List of dictionaries containing post information.
        downloaded_images (list): List of tuples with image name and row index.
        json_filename (str): Name of the JSON file to save.
    """
    json_filename = os.path.abspath(json_filename)  # Ensure it's an absolute path
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)  # Create directory if needed

    formatted_data = []

    for image_name, row_index in downloaded_images:
        if row_index < len(data):  # Ensure index is within range
            entry = data[row_index]

            formatted_entry = {
                "id": image_name.split('.')[0],  
                "image": image_name,  
                "true": 1 if entry.get('headline_label') == 'True' else 0,  
                "headline": entry.get('headline', ''),  # Renamed from 'content' to 'headline'
                "content": entry.get('content', ''),  # Stores first three paragraphs
                "tags": entry.get('tags', ''), # stores article tags
                "source": entry.get('source', 'Unknown'),  
                "more": entry.get('more', ''),  
                "url": entry.get('headline_link', ''), 
            }

            formatted_data.append(formatted_entry)

    with open(json_filename, 'w') as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Data saved to {json_filename}")

def main(num_headlines, num_paragraphs):
    """
    Main function to scrape Politifact data, extract images, and save results.

    - Initialize the scraper with the specified number of headlines.
    - Save the scraped results to a pickle file.
    - Load the pickle file, extract images and article text.
    - Save the final data as a JSON file.

    Args:
        num_headlines (int): Total number of headlines to scrape.
        num_paragraphs (int): Number of paragraphs to extract per article.

    This function will store the final JSON data in './data/news.json' by default.
    """

    scraper = PolitifactScraper(num_headlines=num_headlines)
    scraper.save_results(file_path='data/results.pickle')

    # Load the data from the saved pickle
    with open('data/results.pickle', 'rb') as file:
        data = pickle.load(file)

    # Remove duplicates based on the headline text
    seen_headlines = set()  # Set to track unique headlines
    unique_data = []
    for dic in data:
        headline = dic.get('headline_text')
        if headline and headline not in seen_headlines:
            unique_data.append(dic)
            seen_headlines.add(headline)

    # Extract images and tags
    extractor = ImageExtractor()
    image_text_data = extractor.extract_images_and_text(unique_data, num_paragraphs)  # List of dicts
    extractor.close()

    # Add extracted data to unique_data
    filtered_data = []
    for i, entry in enumerate(image_text_data):
        article_text = entry.get('article_text', [])  # Default to empty list if None
        tags = entry.get('tags', [])  # Extracted tags list

        # Ensure we have valid article text before adding the entry
        if article_text:  
            unique_data[i]['image_url'] = entry.get('image_url', '')  # Add image URL
            unique_data[i]['headline'] = entry.pop('headline_text', '')  # Rename 'headline_text' to 'headline'
            unique_data[i]['content'] = ' '.join(article_text[:num_paragraphs])  # Join first 3 paragraphs
            unique_data[i]['tags'] = tags  # Add extracted tags
            filtered_data.append(unique_data[i])  # Only add valid entries

    # Split data by label
    true_entries = [dic for dic in filtered_data if dic['headline_label'] == 'True']
    false_entries = [dic for dic in filtered_data if dic['headline_label'] == 'False']

    # Find the minimum count to balance classes
    min_count = min(len(true_entries), len(false_entries))

    # Randomly sample min_count elements from both classes
    balanced_true = random.sample(true_entries, min_count)
    balanced_false = random.sample(false_entries, min_count)
    balanced_data = balanced_true + balanced_false

    # Save the balanced dataset
    with open('data/results.pickle', 'wb') as file:
        pickle.dump(balanced_data, file)

    print(f"Filtered and balanced dataset saved with {min_count} True and {min_count} False labels.")
    return balanced_data

# Execution
num_headlines = 1500
num_paragraphs = 4
if __name__ == '__main__':
    # Generate the main data (headlines, paragraphs, etc.)
    data = main(num_headlines=num_headlines, num_paragraphs=num_paragraphs)
     # Download images associated with the data
    downloaded_images = download_images(data, folder='./imgs')
     # Save the data and images info as a JSON file
    save_data_as_json(data, downloaded_images)