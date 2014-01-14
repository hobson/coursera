# Scrapy settings for wikiscrapy project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'wikiscrapy'

SPIDER_MODULES = ['wikiscrapy.spiders']
NEWSPIDER_MODULE = 'wikiscrapy.spiders'

#ROBOTSTXT_OBEY = False

#DEPTH_LIMIT = 0
#DEPTH_STATS = True
#DEPTH_PRIORITY = 0

#DUPEFILTER_CLASS = 'scrapy.dupefilter.RFPDupeFilter'

#CONCURRENT_REQUESTS = 8  # 16
#CONCURRENT_REQUESTS_PER_DOMAIN = 4  # 8

#DOWNLOAD_DELAY = 1.1  # in seconds (requests will be delayed by a minimum of this number of seconds)
#RANDOMIZE_DOWNLOAD_DELAY = True

#AUTOTHROTTLE_START_DELAY = 2.3  # will also obey the DOWNLOAD_DELAY setting
#AUTOTHROTTLE_MAX_DELAY = 10.3

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'SharpLearnerBot/1.0 (+http://www.sharplabs.com)'
