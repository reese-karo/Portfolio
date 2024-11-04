import os
from dotenv import load_dotenv
from apify_client import ApifyClient
from serpapi import GoogleSearch
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
import streamlit as st

class Actor:
    def __init__(self):
        self.serp_run_input = {}
        self.crawl_run_input = {}
        self.url_list = []
        load_dotenv()
        self.apify_api_token = os.getenv('APIFY_API_TOKEN')
        self.client = ApifyClient(self.apify_api_token)
        self.serp_api_token = os.getenv('SERP_API_TOKEN')
        
    def google_search(self, question):
        params = {
        "api_key": self.serp_api_token,
        "engine": "google",
        "q": question,
        "location": "United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": "3",
        "filter": "0",
        "start": "0"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results["organic_results"]
    
    #set crawl input with url list
    def crawler_input(self):
        self.crawl_run_input = {
            "startUrls": [],
            "useSitemaps": False,
            "crawlerType": "playwright:firefox",
            "includeUrlGlobs": [],
            "excludeUrlGlobs": [],
            "ignoreCanonicalUrl": False,
            "maxCrawlDepth": 3,
            "maxCrawlPages": 3,
            "initialConcurrency": 0,
            "maxConcurrency": 200,
            "initialCookies": [],
            "proxyConfiguration": {"useApifyProxy": True},
            "maxSessionRotations": 10,
            "maxRequestRetries": 3,
            "requestTimeoutSecs": 60,
            "dynamicContentWaitSecs": 10,
            "maxScrollHeightPixels": 5000,
            "removeElementsCssSelector": """nav, footer, script, style, noscript, svg,
        [role="alert"],
        [role="banner"],
        [role="dialog"],
        [role="alertdialog"],
        [role="region"][aria-label*="skip" i],
        [aria-modal="true"]""",
            "removeCookieWarnings": True,
            "clickElementsCssSelector": '[aria-expanded="false"]',
            "htmlTransformer": "readableText",
            "readableTextCharThreshold": 100,
            "aggressivePrune": False,
            "debugMode": False,
            "debugLog": False,
            "saveHtml": False,
            "saveMarkdown": True,
            "saveFiles": False,
            "saveScreenshots": False,
            "maxResults": 9999999,
            "clientSideMinChangePercentage": 15,
            "renderingTypeDetectionPercentage": 10,
        }
        for url in self.url_list:
            self.crawl_run_input["startUrls"].append({"url": url})
    
    def run_crawl_loader(self):
        data = self.run_crawl()
        loader = ApifyDatasetLoader(
        dataset_id= data["defaultDatasetId"],
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
        )
        )
        

def main():
    crawler = Actor()
    # Web app title
    st.title('DataScout')

    # Input search query
    user_input = st.text_input("Enter your search query", value="", max_chars=50)

    # Button to initiate search
    if st.button('Search'):
        results = crawler.google_search(user_input)
        for result in results:
            #crawler.url_list.append(result['link'])
            st.subheader(result['title'])
            st.write(result['link'])

# run the app
if __name__ == "__main__":
    #main()
    x = Actor()
    results = x.google_search(input("Enter Topic: "))
    for result in results:
        x.url_list.append(result['link'])
    #print(x.url_list)
    #sets crawler input with links given
    x.crawler_input()
    client = ApifyClient(x.apify_api_token)
    data = client.actor("aYG0l9s7dbB7j3gbS").call(run_input=x.crawl_run_input)
    loader = ApifyDatasetLoader(
        dataset_id= data["defaultDatasetId"],
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
        )
    )
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = input("Enter a question about the topic: ")
    result = index.query_with_sources(query)
    print(result["answer"])
    print(result["sources"])
    

