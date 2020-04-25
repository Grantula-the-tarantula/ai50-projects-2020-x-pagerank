import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    linked = corpus[page]    
    # For pages with no outgoing links, returns distribution of all pages (including itself)
    if len(linked) == 0:
        linked = list(corpus.keys())
        
    # Initializes model as a dictionary
    model = dict()
    
    # Constants
    d = damping_factor / len(linked)
    one_minus_d = (1 - damping_factor) / len(corpus)
    
    # Looping through pages and assigning distribution
    for page in corpus:
        model[page] = one_minus_d
        
        if page in linked:
            model[page] += d
            
    return model       

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # First state is chosen at random
    state = random.choice(list(corpus.keys()))
    pagerank = dict()   
    
    # Sample n times, passing in previous state to transition model and choosing next page
    for i in range(n):
        model = transition_model(corpus, state, damping_factor)
        state = random.choices(list(model.keys()), weights=model.values(), k=1)[0]
        pagerank[state] = pagerank.get(state, 0) + 1
        
    total = sum(pagerank.values())
    pagerank = {key: value / total for key, value in pagerank.items()}
        
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # Initialize constants and lists
    converge = 0.001
    diff = 1
    
    N = len(corpus)
    d = damping_factor
    total_page = list(corpus.keys())
    
    # Set all intial values of pagerank
    pagerank = {k: 1 / N for k in corpus.keys()}
    
    # Loops until the values converge
    while (diff > converge):
        # Loops through pages and calculates first section of PageRank equation
        for page in total_page:
            past_step = pagerank[page]
            pagerank[page] = ((1 - d) / N)
            
            # For each page, loops through links to that page and calculates second section of PageRank equation
            for i in total_page:
                sum_i = 0
                p = corpus[i]
                
                if len(page) == 0:
                    p = total_page
                    
                if page in p:
                    sum_i += pagerank[i] / len(p)                
                pagerank[page] += (d * sum_i)            
            
            # Calculates the different between past step and current value (checks for convergence)
            diff = abs(pagerank[page] - past_step)
            
    return pagerank

if __name__ == "__main__":
    main()

















