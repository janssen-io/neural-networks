class FeatureExtractor:
    def __init__(self, comment=None, keywords=[], domains=[], mean_kw=1, mean_len=1):
        self.comment = comment
        self.domains = domains
        self.keywords = keywords
        self.mean = {'keywords': mean_kw, 'length': mean_len}
        self.weights = {
            'categories': 0.7,
            'keywords': 9.4,
            'length': -1.9,
            'link': 0.7,
            'title': -0.8,
        }

    def extract(self, comment=None):
        self.comment = comment or self.comment
        features = {
            'categories': self.count_categories(),
            'keywords': self.count_keywords(),
            'length':self.count_words(),
            'link': self.link_domain(),
            'title': self.submission_title(),
        }
        return features

    def count_keywords(self):
        count = 0
        text = self.comment.body.lower()
        for word in text.split():
            if word in self.keywords:
                count += 1
        return (1 - ((self.mean['keywords'] - count) / self.mean['keywords'])) / 2

    def count_categories(self):
        count = 0
        cats = ['nose', 'palate', 'finish', 'score', 'color', 'colour', 'abv', 'taste', '/100']
        for cat in cats:
            if cat in self.comment.body.lower():
                count += 1
        return count / 6

    def count_words(self):
        text = self.comment.body
        length = len(text.split())
        return (1 - ((self.mean['length'] - length) / self.mean['length'])) / 2

    def link_domain(self):
        submission = self.comment.submission
        if submission.author != self.comment.author:
            return False

        url = submission.url.lower()
        contains = False
        for domain in self.domains:
            if domain in url:
                contains = True
        return int(contains) / 2

    def submission_title(self):
        submission = self.comment.submission
        title = submission.title.lower()
        return int('review' in title) / 2

# Quick clones the needed properties of PRAW objects
class Submission:
    def __init__(self, author, url, title):
        self.author = author
        self.url = url
        self.title = title

class Comment:
    def __init__(self, cid, author, body, submission):
        self.submission = submission
        self.author = author
        self.body = body
        self.id = cid
