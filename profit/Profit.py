def getActionProfit(buy_price, sell_price):
    if(sell_price >= buy_price):
        percentage_gain = ((sell_price-buy_price)/buy_price)*100
        return "BUY"
    else:
        return "CONTINUE"
