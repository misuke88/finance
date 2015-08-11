# 8-k report analysis #

1. 8k report를 바탕으로 sentiment를 분석하고자 한다.
	- **unigram**을 통해 term vector를 얻고 이 vector를 주식 예측 모델에 사용하고자 함
	- term을 바탕으로 sentiment (domain: finance) dictionary를 만들기

## Requirments

- python 2.7+
- lee2014: http://nlp.stanford.edu/pubs/stock-event.html

## Configuration

    cp settings.py.sample settings.py

## Run

1. `parsing.py`: Creates `stock.txt`
1. `preprocessing_for_response.py`: Creates `stock_{X,y}.txt`
1. `modeling.py`: Class prediction

or

    make run

## Author

- [Misuk Kim](http://github.com/misuke88)
- [Lucy Park](http://github.com/e9t)
