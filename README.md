# Stock prediction

## Data

- Source: http://nlp.stanford.edu/pubs/stock-event.html

### 8K-gz
- Format: text in gz
    - delimited with `<document>` tags
- Filenames: company codes
- Features
    - FILE
    - TIME: time stamp
    - EVENTS: tags
    - ITEM

### EPS
- Format: html in txt
- Filenames: dates
- Parsed version: `EPS/eps.ser`

### price_history
- Format: csv
- Filenames: company codes
- Features: Date, Open, High, Low, Close, Volume, Adj Close
- Meta data
    - `djia.csv`: 다우존스 industrial average
    - `gspc.csv`: SNP 500
    - `ixic.csv`: NASDAQ composite
    - `vix.csv`: CBOE Volatility SNP 500

### snp_list
- Format: tsv in txt
- Content: Company codes and company categories
