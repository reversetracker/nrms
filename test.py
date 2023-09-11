import pandas as pd

# CSV 파일 로드
df = pd.read_csv('/Users/nelly/TrollProjects/nrms/bq-results-20230901.csv')

# 첫 20000개의 행만 선택
df = df.iloc[:20000]

# 동일한 파일에 다시 저장
df.to_csv('/Users/nelly/TrollProjects/nrms/bq-results-20230902.csv', index=False)
