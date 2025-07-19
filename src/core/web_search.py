import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from src.utils.logger import get_logger
from src.config.settings import settings
import re

logger = get_logger()

class WebSearchService:
    def __init__(self):
        self.enabled = settings.enable_web_search
        self.max_results = settings.web_search_results
        self.python_keywords = [
            'python', 'programming', 'code', 'syntax', 'tutorial',
            'documentation', 'function', 'class', 'method', 'variable',
            'list', 'dict', 'string', 'integer', 'loop', 'conditional'
        ]
    
    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search web using DuckDuckGo with improved Python-specific filtering"""
        if not self.enabled:
            return []
        
        try:
            results = []
            
            # Enhanced search query for Python-specific content
            enhanced_query = f'"{query}" python programming tutorial documentation -medical -health -clinic'
            
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    enhanced_query,
                    max_results=self.max_results * 2,  # Get more to filter better
                    region='wt-wt'  # Worldwide
                )
                
                for result in search_results:
                    # Filter for Python-relevant content
                    if self._is_python_relevant(result):
                        content = self._extract_content(result['href'])
                        
                        if content and self._is_content_python_relevant(content):
                            results.append({
                                'title': result['title'],
                                'url': result['href'],
                                'snippet': result['body'],
                                'content': content[:1500],  # Limit content length
                                'source': 'web_search',
                                'relevance_score': self._calculate_python_relevance(result, content)
                            })
                    
                    if len(results) >= self.max_results:
                        break
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Web search returned {len(results)} relevant Python results")
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def _is_python_relevant(self, result: Dict) -> bool:
        """Check if search result is Python-relevant"""
        text_to_check = f"{result['title']} {result['body']}".lower()
        
        # Must contain 'python' and at least one programming keyword
        has_python = 'python' in text_to_check
        has_programming_keyword = any(keyword in text_to_check for keyword in self.python_keywords[1:])
        
        # Exclude medical/health/irrelevant domains
        excluded_domains = ['mayoclinic.org', 'webmd.com', 'healthline.com', 'medical', 'health']
        is_excluded = any(domain in result['href'].lower() for domain in excluded_domains)
        
        # Prefer certain domains
        preferred_domains = ['python.org', 'docs.python.org', 'stackoverflow.com', 'realpython.com', 
                           'geeksforgeeks.org', 'w3schools.com', 'tutorialspoint.com']
        is_preferred = any(domain in result['href'].lower() for domain in preferred_domains)
        
        return (has_python and has_programming_keyword and not is_excluded) or is_preferred
    
    def _is_content_python_relevant(self, content: str) -> bool:
        """Check if extracted content is Python-relevant"""
        content_lower = content.lower()
        
        # Count Python-related terms
        python_terms = sum(1 for keyword in self.python_keywords if keyword in content_lower)
        
        # Check for code patterns
        has_code_patterns = bool(re.search(r'(def |class |import |from |print\(|\.py)', content))
        
        return python_terms >= 3 or has_code_patterns
    
    def _calculate_python_relevance(self, result: Dict, content: str) -> float:
        """Calculate relevance score for Python content"""
        score = 0.0
        
        text_to_check = f"{result['title']} {result['body']} {content}".lower()
        
        # Domain scoring
        preferred_domains = {
            'docs.python.org': 1.0,
            'python.org': 0.9,
            'stackoverflow.com': 0.8,
            'realpython.com': 0.8,
            'geeksforgeeks.org': 0.7,
            'w3schools.com': 0.6
        }
        
        for domain, domain_score in preferred_domains.items():
            if domain in result['href']:
                score += domain_score
                break
        else:
            score += 0.3  # Default score for other domains
        
        # Keyword frequency scoring
        keyword_count = sum(text_to_check.count(keyword) for keyword in self.python_keywords)
        score += min(keyword_count * 0.1, 0.5)  # Max 0.5 from keywords
        
        # Code pattern bonus
        if re.search(r'(def |class |import |from |print\()', content):
            score += 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_content(self, url: str) -> str:
        """Extract content from a URL with better filtering"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""