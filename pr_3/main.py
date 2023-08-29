import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px


def z_1_2():
    energy = pd.read_excel('energy.xlsx', sheet_name='L1')
    goldprices = pd.read_csv('goldprices.csv', sep=';')
    walmart_Store_Sales = pd.read_csv('Walmart_Store_sales.csv', sep=';')

    print('---ENERGY---\n---INFO---')
    energy.info()
    print(' ---HEAD--- \n', energy.head())
    print('---INSA.SUM---\n', energy.isna().sum())

    print('\n---GOLDPRICES---\n---INFO---')
    goldprices.info()
    print(' ---HEAD--- \n', goldprices.head())
    print('---INSA.SUM---\n', goldprices.isna().sum())

    print('\n---WALMART_STORE_SALES---\n---INFO---')
    walmart_Store_Sales.info()
    print(' ---HEAD--- \n', walmart_Store_Sales.head())
    print('---INSA.SUM---\n', walmart_Store_Sales.isna().sum())


def z_3():
    energy = pd.read_excel('energy.xlsx', sheet_name='L1')

    # fig = go.Figure(px.bar(x=energy["Year"], y=energy["Consumption"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=energy["Year"], y=energy["Consumption"]))
    fig.update_traces(marker=dict(color=energy["Year"], coloraxis="coloraxis"))
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.update_layout(title={'text': 'Диаграмма Energy', 'font_size': 20, 'y': 0.96, 'x': 0.55},
                      height=700,
                      xaxis_title='Year', xaxis_title_font_size=16, xaxis_tickfont_size=14, xaxis_tickangle=315,
                      yaxis_title='Consumption', yaxis_title_font_size=16, yaxis_tickfont_size=14,
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.show()


def z_4():
    energy = pd.read_excel('energy.xlsx', sheet_name='L1')

    fig = go.Figure()
    fig.add_trace(go.Pie(values=energy["Consumption"], labels=energy["Year"], hole=0.9))
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.update_layout(annotations=[dict(text='Диаграмма Energy', x=0.5, y=0.5, font_size=20, showarrow=False)],
                      height=700,
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.show()


def z_5():
    energy = pd.read_excel('energy.xlsx', sheet_name='L1')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy["Year"], y=energy["Consumption"], mode='lines+markers'))
    fig.update_traces(marker=dict(color='white', line=dict(color='black', width=2)))
    fig.update_traces(line=dict(color='crimson'))
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()


if __name__ == '__main__':
    z = int(input('Выберите задание: '))
    if z == 1 or z == 2:
        z_1_2()
    elif z == 3:
        z_3()
    elif z == 4:
        z_4()
    elif z == 5:
        z_5()
    else:
        print("Доступные задание [;]")
