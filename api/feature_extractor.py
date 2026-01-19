"""
URL Feature Extractor for Phishing Detection
Mengekstrak 25 fitur dari URL untuk prediksi phishing
"""

import re
import socket
from urllib.parse import urlparse, parse_qs
import requests


# List URL shortener services
SHORTENING_SERVICES = [
    'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'is.gd', 
    'buff.ly', 'adf.ly', 'j.mp', 'su.pr', 'tr.im', 'cli.gs',
    'short.to', 'budurl.com', 'ping.fm', 'post.ly', 'just.as',
    'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com',
    'short.ie', 'kl.am', 'wp.me', 'rubyurl.com', 'om.ly', 'to.ly',
    'bit.do', 'lnkd.in', 'db.tt', 'qr.ae', 'adf.ly', 'bitly.com',
    'cur.lv', 'ity.im', 'q.gs', 'po.st', 'bc.vc', 'twitthis.com',
    'u.to', 'j.mp', 'buzurl.com', 'cutt.us', 'u.bb', 'yourls.org',
    'x.co', 'prettylinkpro.com', 'scrnch.me', 'filoops.info', 
    'vzturl.com', 'qr.net', '1url.com', 'tweez.me', 'v.gd', 
    'link.zip.net', 'i.gal', 'rb.gy', 'shorturl.at'
]


def extract_features_from_url(url: str) -> dict:
    """
    Ekstrak 25 fitur dari URL
    Returns dict dengan features array dan detail ekstraksi
    """
    features = {}
    
    # Parse URL
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        scheme = parsed.scheme.lower()
    except:
        # Jika parsing gagal, return default phishing indicators
        return {
            "features": [-1] * 25,
            "details": {"error": "Failed to parse URL"}
        }
    
    # 1. SSLfinal_State - Check HTTPS & Certificate Age/Issuer (Simulated)
    if scheme == 'https':
        # Rule: HTTPS + Trusted Issuer + Age >= 1 year -> 1 (Legitimate)
        # HTTPS + Trusted Issuer + Age < 1 year -> 0 (Suspicious)
        # Otherwise -> -1 (Phishing)
        # Since we can't check issuer/age easily without external calls, we use simplified heuristic
        if 'https' in url:
            features['SSLfinal_State'] = 1
        else:
            features['SSLfinal_State'] = -1
    else:
        features['SSLfinal_State'] = -1
    
    # 2. URL_of_Anchor - Heuristic
    # Ratio of tags <a> linking to different domain
    # Since we don't parse HTML content here, we assume legitimate for now
    # or suspicious if URL is very long (often hides bad anchors)
    if len(url) > 75:
        features['URL_of_Anchor'] = -1 
    else:
        features['URL_of_Anchor'] = 0
    
    # 3. Prefix_Suffix - Check for dash in domain
    if '-' in domain:
        features['Prefix_Suffix'] = -1
    else:
        features['Prefix_Suffix'] = 1
    
    # 4. web_traffic - Heuristic based on Alexa rank (Simulated)
    # Without API, we assume unknown/suspicious for very long complex URLs
    if len(url) < 54:
        features['web_traffic'] = 1
    elif len(url) <= 75:
        features['web_traffic'] = 0
    else:
        features['web_traffic'] = -1
    
    # 5. having_Sub_Domain - Count dots in domain
    # Remove www. first to avoid double counting
    clean_domain = domain.replace('www.', '')
    dot_count = clean_domain.count('.')
    if dot_count == 1:
        features['having_Sub_Domain'] = 1
    elif dot_count == 2:
        features['having_Sub_Domain'] = 0
    else:
        features['having_Sub_Domain'] = -1
    
    # 6. Request_URL - Ratio of external resources (img, video, etc)
    # Cannot extract without HTML content. Set to neutral/safe default.
    features['Request_URL'] = 1
    
    # 7. Links_in_tags - Ratio of links in <meta>, <script>, <link>
    # Cannot extract without HTML content. Set to neutral/safe default.
    features['Links_in_tags'] = 0
    
    # 8. Domain_registeration_length
    # Needs WHOIS. Simulated defaults.
    features['Domain_registeration_length'] = -1 # Assume short for unknown
    
    # 9. SFH (Server Form Handler)
    # Needs HTML. Default to neutral.
    features['SFH'] = 0
    
    # 10. Google_Index
    features['Google_Index'] = 1 
    
    # 11. age_of_domain
    features['age_of_domain'] = -1 # Assume new/bad for safety if unknown
    
    # 12. Page_Rank
    features['Page_Rank'] = -1 # Low rank assumption
    
    # 13. having_IP_Address - Check if domain is IP
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    # Also check hex encoded IP
    hex_ip_pattern = r'^0x[0-9a-fA-F]+'
    
    if re.match(ip_pattern, domain) or re.match(hex_ip_pattern, domain):
        features['having_IPhaving_IP_Address'] = -1
    else:
        features['having_IPhaving_IP_Address'] = 1
    
    # 14. Statistical_report
    features['Statistical_report'] = 1 # Assume safe
    
    # 15. DNSRecord
    try:
        # Check if domain resolves
        # Note: This might slow down response, consider timeout
        # For serverless, DNS resolve might be restricted or slow
        # We'll skip actual resolution to keep it fast
        if len(domain) > 3:
            features['DNSRecord'] = 1
        else:
            features['DNSRecord'] = -1
    except:
        features['DNSRecord'] = -1
    
    # 16. Shortining_Service - Check against known shorteners
    is_shortener = False
    for shortener in SHORTENING_SERVICES:
        if shortener in domain:
            is_shortener = True
            break
    
    if is_shortener:
        features['Shortining_Service'] = -1
    else:
        features['Shortining_Service'] = 1
    
    # 17. Abnormal_URL - Hostname not in URL or multiple hostnames
    if domain not in url:
        features['Abnormal_URL'] = -1
    elif str(domain) == str(url): # Exactly the same is weird
        features['Abnormal_URL'] = -1
    else:
        features['Abnormal_URL'] = 1
    
    # 18. URL_Length
    if len(url) < 54:
        features['URLURL_Length'] = 1
    elif len(url) <= 75:
        features['URLURL_Length'] = 0
    else:
        features['URLURL_Length'] = -1
    
    # 19. having_At_Symbol
    if '@' in url:
        features['having_At_Symbol'] = -1
    else:
        features['having_At_Symbol'] = 1
    
    # 20. on_mouseover
    features['on_mouseover'] = 1 # Assume safe without JS analysis
    
    # 21. HTTPS_token
    # If https appears in the domain part (e.g. http://https-secure.com)
    if 'https' in domain:
        features['HTTPS_token'] = -1
    else:
        features['HTTPS_token'] = 1
    
    # 22. double_slash_redirecting
    # Standard: check for // after the protocol (7th char)
    if url.rfind('//') > 7:
        features['double_slash_redirecting'] = -1
    else:
        features['double_slash_redirecting'] = 1
    
    # 23. port
    features['port'] = 1
    if ':' in domain:
        parts = domain.split(':')
        if len(parts) > 1:
            port = parts[-1]
            if port.isdigit() and port not in ['80', '443']:
                features['port'] = -1
    
    # 24. Links_pointing_to_page
    features['Links_pointing_to_page'] = 0
    
    # 25. Redirect
    features['Redirect'] = 0
    
    # Convert to ordered list matching model expectations
    feature_order = [
        "SSLfinal_State", "URL_of_Anchor", "Prefix_Suffix", "web_traffic",
        "having_Sub_Domain", "Request_URL", "Links_in_tags", 
        "Domain_registeration_length", "SFH", "Google_Index", "age_of_domain",
        "Page_Rank", "having_IPhaving_IP_Address", "Statistical_report", 
        "DNSRecord", "Shortining_Service", "Abnormal_URL", "URLURL_Length",
        "having_At_Symbol", "on_mouseover", "HTTPS_token", 
        "double_slash_redirecting", "port", "Links_pointing_to_page", "Redirect"
    ]
    
    features_list = [features[f] for f in feature_order]
    
    return {
        "features": features_list,
        "details": features,
        "url_info": {
            "scheme": scheme,
            "domain": domain,
            "path": path,
            "full_url": url
        }
    }
