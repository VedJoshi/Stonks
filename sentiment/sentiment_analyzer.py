import praw
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import re

class SentimentAnalyzer:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        self.vader = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Sentiment keywords for enhanced analysis
        self.bullish_keywords = [
            'moon', 'rocket', 'bull', 'bullish', 'buy', 'long', 'calls', 'up', 'rise',
            'pump', 'green', 'gains', 'profit', 'buy the dip', 'diamond hands', 'hodl',
            'to the moon', 'stonks', 'yolo', 'ath', 'breakout', 'rally'
        ]
        
        self.bearish_keywords = [
            'crash', 'bear', 'bearish', 'sell', 'short', 'puts', 'down', 'fall',
            'dump', 'red', 'loss', 'rip', 'dead cat bounce', 'paper hands',
            'bubble', 'overvalued', 'recession', 'correction', 'support level'
        ]
    
    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using multiple methods"""
        if not text:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        cleaned_text = self.clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        
        # Keyword-based sentiment boost
        keyword_sentiment = self.analyze_keywords(cleaned_text)
        
        # Combine scores with weights
        combined_sentiment = (
            vader_scores['compound'] * 0.5 +
            textblob_polarity * 0.3 +
            keyword_sentiment * 0.2
        )
        
        # Calculate confidence based on agreement between methods
        sentiment_agreement = abs(vader_scores['compound'] - textblob_polarity)
        confidence = max(0.3, 1.0 - sentiment_agreement)
        
        return {
            'sentiment': combined_sentiment,
            'confidence': confidence,
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': textblob_polarity,
            'keyword_sentiment': keyword_sentiment
        }
    
    def analyze_keywords(self, text):
        """Analyze sentiment based on specific keywords"""
        if not text:
            return 0.0
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
        
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0
        
        # Return sentiment score between -1 and 1
        return (bullish_count - bearish_count) / total_keywords
    
    def get_reddit_sentiment(self, symbol, limit=50):
        """Get sentiment from Reddit for a specific symbol"""
        try:
            sentiments = []
            post_count = 0
            
            # Search multiple subreddits
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts about the symbol
                    for submission in subreddit.search(symbol, time_filter='day', limit=limit//len(subreddits)):
                        # Analyze title and selftext
                        full_text = f"{submission.title} {submission.selftext}"
                        sentiment_result = self.analyze_text_sentiment(full_text)
                        
                        if sentiment_result['confidence'] > 0.5:
                            sentiments.append({
                                'sentiment': sentiment_result['sentiment'],
                                'confidence': sentiment_result['confidence'],
                                'upvotes': submission.score,
                                'comments': submission.num_comments,
                                'text': full_text[:200]  # Store first 200 chars
                            })
                            post_count += 1
                        
                        time.sleep(1)  # Rate limiting
                
                except Exception as e:
                    self.logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            if not sentiments:
                return {
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'post_count': 0,
                    'source': 'reddit'
                }
            
            # Calculate weighted average sentiment
            total_weight = 0
            weighted_sentiment = 0
            
            for item in sentiments:
                # Weight by upvotes and confidence
                weight = (item['upvotes'] + 1) * item['confidence']
                weighted_sentiment += item['sentiment'] * weight
                total_weight += weight
            
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = sum(item['confidence'] for item in sentiments) / len(sentiments)
            
            return {
                'sentiment': avg_sentiment,
                'confidence': avg_confidence,
                'post_count': post_count,
                'source': 'reddit'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'post_count': 0, 'source': 'reddit'}
    
    def get_news_sentiment(self, symbol):
        """Get sentiment from news sources (simplified implementation)"""
        try:
            # This is a simplified implementation - in production you'd use real news APIs
            # like Alpha Vantage, News API, or Financial Modeling Prep
            
            # For now, return neutral sentiment
            return {
                'sentiment': 0.0,
                'confidence': 0.3,
                'article_count': 0,
                'source': 'news'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0, 'source': 'news'}
    
    def get_twitter_sentiment(self, symbol):
        """Get sentiment from Twitter (placeholder implementation)"""
        try:
            # This would require Twitter API v2 setup
            # For now, return neutral sentiment
            
            return {
                'sentiment': 0.0,
                'confidence': 0.2,
                'tweet_count': 0,
                'source': 'twitter'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'tweet_count': 0, 'source': 'twitter'}
    
    def get_combined_sentiment(self, symbol):
        """Get combined sentiment from all sources"""
        try:
            self.logger.info(f"Getting combined sentiment for {symbol}")
            
            # Get sentiment from all sources
            reddit_sentiment = self.get_reddit_sentiment(symbol)
            news_sentiment = self.get_news_sentiment(symbol)
            twitter_sentiment = self.get_twitter_sentiment(symbol)
            
            sources = [reddit_sentiment, news_sentiment, twitter_sentiment]
            
            # Filter out sources with zero confidence
            valid_sources = [s for s in sources if s['confidence'] > 0]
            
            if not valid_sources:
                return {
                    'combined_sentiment': 0.0,
                    'combined_confidence': 0.0,
                    'reddit_sentiment': reddit_sentiment['sentiment'],
                    'reddit_confidence': reddit_sentiment['confidence'],
                    'reddit_posts': reddit_sentiment['post_count'],
                    'news_sentiment': news_sentiment['sentiment'],
                    'news_confidence': news_sentiment['confidence'],
                    'twitter_sentiment': twitter_sentiment['sentiment'],
                    'twitter_confidence': twitter_sentiment['confidence']
                }
            
            # Calculate weighted average
            total_weight = sum(s['confidence'] for s in valid_sources)
            combined_sentiment = sum(s['sentiment'] * s['confidence'] for s in valid_sources) / total_weight
            combined_confidence = total_weight / len(sources)  # Average confidence
            
            result = {
                'combined_sentiment': combined_sentiment,
                'combined_confidence': combined_confidence,
                'reddit_sentiment': reddit_sentiment['sentiment'],
                'reddit_confidence': reddit_sentiment['confidence'],
                'reddit_posts': reddit_sentiment['post_count'],
                'news_sentiment': news_sentiment['sentiment'],
                'news_confidence': news_sentiment['confidence'],
                'twitter_sentiment': twitter_sentiment['sentiment'],
                'twitter_confidence': twitter_sentiment['confidence']
            }
            
            self.logger.info(f"Combined sentiment for {symbol}: "
                           f"Sentiment={combined_sentiment:.3f}, "
                           f"Confidence={combined_confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting combined sentiment for {symbol}: {e}")
            return {
                'combined_sentiment': 0.0,
                'combined_confidence': 0.0,
                'reddit_sentiment': 0.0,
                'reddit_confidence': 0.0,
                'reddit_posts': 0,
                'news_sentiment': 0.0,
                'news_confidence': 0.0,
                'twitter_sentiment': 0.0,
                'twitter_confidence': 0.0
            }
    
    def validate_sentiment_quality(self, sentiment_data):
        """Validate sentiment data quality"""
        if not sentiment_data:
            return False
        
        # Check if we have sufficient confidence
        if sentiment_data.get('combined_confidence', 0) < 0.4:
            return False
        
        # Check if we have sufficient data sources
        sources_with_data = 0
        if sentiment_data.get('reddit_confidence', 0) > 0.3:
            sources_with_data += 1
        if sentiment_data.get('news_confidence', 0) > 0.3:
            sources_with_data += 1
        if sentiment_data.get('twitter_confidence', 0) > 0.3:
            sources_with_data += 1
        
        return sources_with_data >= 1  # At least one good source
        """Analyze sentiment of text using multiple methods"""
        if not text:
            return {'compound': 0, 'confidence': 0}
        
        cleaned_text = self.clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        
        # Keyword-based sentiment boost
        keyword_sentiment = self.analyze_keywords(cleaned_text)
        
        # Combine scores with weights
        combined_sentiment = (
            vader_scores['compound'] * 0.5 +
            textblob_polarity * 0.3 +
            keyword_sentiment * 0.2
        )
        
        # Calculate confidence based on agreement between methods
        sentiment_agreement = abs(vader_scores['compound'] - textblob_polarity)
        confidence = max(0.3, 1.0 - sentiment_agreement)
        
        return {
            'compound': combined_sentiment,
            'confidence': confidence,
            'vader': vader_scores,
            'textblob': textblob_polarity,
            'keyword': keyword_sentiment
        }
    
    def analyze_keywords(self, text):
        """Analyze sentiment based on financial keywords"""
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
        
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0
        
        return (bullish_count - bearish_count) / total_keywords
    
    def get_reddit_sentiment(self, symbol, limit=50):
        """Get sentiment from Reddit for a specific symbol"""
        try:
            self.logger.info(f"Fetching Reddit sentiment for {symbol}")
            
            # Search multiple subreddits
            subreddits = ['wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting']
            all_posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the symbol
                    search_terms = [symbol, f"${symbol}", f"{symbol} stock"]
                    
                    for search_term in search_terms:
                        try:
                            posts = subreddit.search(search_term, time_filter='day', limit=limit//len(search_terms))
                            
                            for post in posts:
                                # Check if symbol is actually mentioned in title or text
                                full_text = f"{post.title} {post.selftext}"
                                if symbol.upper() in full_text.upper():
                                    all_posts.append({
                                        'title': post.title,
                                        'text': post.selftext,
                                        'score': post.score,
                                        'num_comments': post.num_comments,
                                        'created_utc': post.created_utc,
                                        'upvote_ratio': post.upvote_ratio,
                                        'subreddit': subreddit_name
                                    })
                        except Exception as e:
                            self.logger.warning(f"Error searching {subreddit_name} for {search_term}: {e}")
                            continue
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            if not all_posts:
                self.logger.warning(f"No Reddit posts found for {symbol}")
                return {
                    'sentiment_score': 0,
                    'confidence': 0,
                    'post_count': 0,
                    'avg_score': 0,
                    'total_comments': 0
                }
            
            # Analyze sentiment for each post
            sentiments = []
            total_score = 0
            total_comments = 0
            
            for post in all_posts:
                full_text = f"{post['title']} {post['text']}"
                sentiment_result = self.analyze_text_sentiment(full_text)
                
                # Weight sentiment by post engagement (score + comments)
                engagement_weight = max(1, post['score'] + post['num_comments'])
                weighted_sentiment = sentiment_result['compound'] * engagement_weight
                
                sentiments.append({
                    'sentiment': sentiment_result['compound'],
                    'confidence': sentiment_result['confidence'],
                    'weight': engagement_weight
                })
                
                total_score += post['score']
                total_comments += post['num_comments']
            
            # Calculate weighted average sentiment
            total_weight = sum(s['weight'] for s in sentiments)
            if total_weight > 0:
                avg_sentiment = sum(s['sentiment'] * s['weight'] for s in sentiments) / total_weight
                avg_confidence = sum(s['confidence'] * s['weight'] for s in sentiments) / total_weight
            else:
                avg_sentiment = 0
                avg_confidence = 0
            
            result = {
                'sentiment_score': avg_sentiment,
                'confidence': avg_confidence,
                'post_count': len(all_posts),
                'avg_score': total_score / len(all_posts) if all_posts else 0,
                'total_comments': total_comments,
                'raw_posts': all_posts[:10]  # Keep sample of posts
            }
            
            self.logger.info(f"Reddit sentiment for {symbol}: {avg_sentiment:.3f} (confidence: {avg_confidence:.3f}, posts: {len(all_posts)})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0,
                'confidence': 0,
                'post_count': 0,
                'avg_score': 0,
                'total_comments': 0,
                'error': str(e)
            }
    
    def get_news_sentiment(self, symbol):
        """Get sentiment from financial news (placeholder for news API)"""
        try:
            # This is a placeholder implementation
            # In production, you would integrate with news APIs like:
            # - Alpha Vantage News API
            # - NewsAPI
            # - Financial Modeling Prep
            # - Polygon.io News
            
            self.logger.info(f"Fetching news sentiment for {symbol} (placeholder)")
            
            # Simulate news sentiment analysis
            # In real implementation, you'd fetch and analyze news articles
            return {
                'sentiment_score': 0,
                'confidence': 0.3,
                'article_count': 0,
                'source': 'news_placeholder'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0,
                'confidence': 0,
                'article_count': 0,
                'error': str(e)
            }
    
    def get_twitter_sentiment(self, symbol):
        """Get sentiment from Twitter (placeholder for Twitter API)"""
        try:
            # This is a placeholder implementation
            # In production, you would integrate with Twitter API v2
            # Note: Twitter API requires approval and has rate limits
            
            self.logger.info(f"Fetching Twitter sentiment for {symbol} (placeholder)")
            
            # Simulate Twitter sentiment analysis
            return {
                'sentiment_score': 0,
                'confidence': 0.3,
                'tweet_count': 0,
                'source': 'twitter_placeholder'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0,
                'confidence': 0,
                'tweet_count': 0,
                'error': str(e)
            }
    
    def get_combined_sentiment(self, symbol):
        """Get combined sentiment from all sources"""
        try:
            self.logger.info(f"Getting combined sentiment analysis for {symbol}")
            
            # Get sentiment from all sources
            reddit_sentiment = self.get_reddit_sentiment(symbol)
            news_sentiment = self.get_news_sentiment(symbol)
            twitter_sentiment = self.get_twitter_sentiment(symbol)
            
            # Weight different sources based on reliability and data availability
            sources = [
                {'data': reddit_sentiment, 'weight': 0.6, 'name': 'reddit'},
                {'data': news_sentiment, 'weight': 0.3, 'name': 'news'},
                {'data': twitter_sentiment, 'weight': 0.1, 'name': 'twitter'}
            ]
            
            # Calculate weighted sentiment
            total_weight = 0
            weighted_sentiment = 0
            weighted_confidence = 0
            
            for source in sources:
                data = source['data']
                weight = source['weight']
                
                # Only include sources with actual data
                if data['confidence'] > 0:
                    weighted_sentiment += data['sentiment_score'] * weight * data['confidence']
                    weighted_confidence += data['confidence'] * weight
                    total_weight += weight * data['confidence']
            
            # Normalize by total weight
            if total_weight > 0:
                final_sentiment = weighted_sentiment / total_weight
                final_confidence = weighted_confidence / sum(s['weight'] for s in sources)
            else:
                final_sentiment = 0
                final_confidence = 0
            
            # Ensure confidence is reasonable
            final_confidence = max(0.1, min(1.0, final_confidence))
            
            result = {
                'combined_sentiment': final_sentiment,
                'combined_confidence': final_confidence,
                'reddit_sentiment': reddit_sentiment['sentiment_score'],
                'reddit_confidence': reddit_sentiment['confidence'],
                'reddit_posts': reddit_sentiment['post_count'],
                'news_sentiment': news_sentiment['sentiment_score'],
                'news_confidence': news_sentiment['confidence'],
                'twitter_sentiment': twitter_sentiment['sentiment_score'],
                'twitter_confidence': twitter_sentiment['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Combined sentiment for {symbol}: {final_sentiment:.3f} (confidence: {final_confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting combined sentiment for {symbol}: {e}")
            return {
                'combined_sentiment': 0,
                'combined_confidence': 0,
                'reddit_sentiment': 0,
                'reddit_confidence': 0,
                'reddit_posts': 0,
                'news_sentiment': 0,
                'news_confidence': 0,
                'twitter_sentiment': 0,
                'twitter_confidence': 0,
                'error': str(e)
            }
    
    def validate_sentiment_quality(self, sentiment_data):
        """Validate the quality of sentiment data"""
        quality_score = 0
        
        # Check data recency
        if 'timestamp' in sentiment_data:
            timestamp = datetime.fromisoformat(sentiment_data['timestamp'].replace('Z', '+00:00'))
            time_diff = datetime.now() - timestamp
            if time_diff < timedelta(hours=1):
                quality_score += 0.3
        
        # Check confidence level
        confidence = sentiment_data.get('combined_confidence', 0)
        if confidence > 0.6:
            quality_score += 0.4
        elif confidence > 0.3:
            quality_score += 0.2
        
        # Check data volume
        reddit_posts = sentiment_data.get('reddit_posts', 0)
        if reddit_posts > 10:
            quality_score += 0.3
        elif reddit_posts > 5:
            quality_score += 0.2
        elif reddit_posts > 0:
            quality_score += 0.1
        
        return quality_score
