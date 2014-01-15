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

ROBOTSTXT_OBEY = True

DEPTH_LIMIT = 0
DEPTH_STATS = True
DEPTH_PRIORITY = 0

DUPEFILTER_CLASS = 'scrapy.dupefilter.RFPDupeFilter'

CONCURRENT_REQUESTS = 1  # 16
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # 8

DOWNLOAD_DELAY = 1.1  # in seconds (requests will be delayed by a minimum of this number of seconds)
RANDOMIZE_DOWNLOAD_DELAY = True

AUTOTHROTTLE_START_DELAY = 2.3  # will also obey the DOWNLOAD_DELAY setting
AUTOTHROTTLE_MAX_DELAY = 10.3

#Crawl responsibly by identifying yourself (and your website) on the user-agent
# but this user agent results in en.m.wikipedia links (mobile)
#USER_AGENT = 'SharpBot/1.0 (+http://www.sharplabs.com)'
#USER_AGENT = 'Mozilla/5.0 (compatible; Sharpbot/1.0; +http://www.sharplabs.com)'
# if you put a url (webpage) in the USER_AGENT string wikipedia redirects you the mobile site and the full text isn't available unless you browse to subsections
USER_AGENT = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:24.0) Gecko/20100101 Firefox/24.0'
