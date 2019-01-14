from datetime import datetime
import fix_yahoo_finance as yf

# Download data from yahoo
def yahooFinanceDownload(ticker):
    '''Download daily stock prices for a single stock from Yahoo! Finance and reserve data as CSV to specific filepath.

    Args:
        ticker(str): Ticker
        filepath(str): Filepath to output and reserve CSV
    '''
    start_date= datetime.datetime(2000,1,1)
    end_date = datetime.date.today()
    prices = yf.download(ticker, start=start_date, end=end_date)
    return prices