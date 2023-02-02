<a href="https://linkedin.com/in/enekoegiguren" target="_blank">
<img src=https://img.shields.io/badge/linkedin-%231E77B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white alt=linkedin style="margin-bottom: 5px;" />
</a>  
</div>

# 🛒 Spanish supermarket price evolution analysis and prediction


![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/presentation.png)

<div align="left">
  
## DATA: The data has been extracted from [DATA Market](https://datamarket.es/#productos-de-supermercados-dataset)

  DataMarket is a company specialized in data, with the purpose of providing organizations with an external layer of information for analysis and decision-making in a much more effective way.
  
<a href="https://datamarket.es">
  <img src="https://datamarket.es/media/banners/productos-de-supermercados-banner.png">
</a>
  
### Another sources of data has been utilizated:
  
* __European Central Bank:__ the [Harmonised Index of Consumer Prices (HICP)](https://www.ecb.europa.eu/stats/macroeconomic_and_sectoral/hicp/html/index.es.html) has been added. Is used to measure consumer price inflation. 
* __INE (Instituto Nacional de Estadistica)__: [CPI](https://www.ine.es/jaxiT3/Tabla.htm?t=50902&L=0) of aliments to take into account inflation in Spain.
  
## EDA:
  
  #### About the dataset:
  
```
df.columns
  
  > Index(['url', 'supermarket', 'category', 'name', 'description', 'price',
       'reference_price', 'reference_unit', 'insert_date', 'product_id',
       'year', 'month', 'day', 'year_month', 'OBS_VALUE_ANR', 'EA_hist_value',
       'indice_ine'],
      dtype='object')
```
```
df.head
```
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/df.head.png)

#### Null values:
  
```
null_values = pd.DataFrame({'Null Values' : df.isna().sum(), '% Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
null_values
```
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/nullvalues.png)
  
## Categories of products:
  
The first consult about the dataset gives us 1504 different categories.
```
df.category.nunique()
  
  > 1504
```
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/category2.png)
  
```
  #If we get the first word of the category we'll have more less categories:

df["categoria"] = df["category"].apply(lambda x: x.split("_")[0])
  
  > We have: 41 different categories
```
  
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/category3.png)
  
## Price:

The first analysis of the price give us the next boxplot and price distribution:
  
<div align="center">
  
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price1.png)
</div>
  
<div align="center">
  
*Boxplot of the price*

  
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price2.png)
</div>
  
<div align="center">

*Distribution and scatter plot*
  
</div>
  
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price3.png)
</div>
  
<div align="center">

*The evolution of the price*
  
</div>

It's obvious that we have outliers that do not allow to analyze the data correctly.
For that we are going to filter the data:
  
```
df1 = df[(df['price'] <=10) & (df['reference_price'] <=200)]


porcentaje = ((df.shape[0]-df1.shape[0])/df.shape[0])
print('We are losing the {:.2f}% of the data\n'.format(porcentaje)
  
  > We are losing the 0.06% of the data
```
**The new distribution of the data and the scatter plot after removing the outliers are:**

![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price4.png)
</div>
  
<div align="center">

*The distribution and scatter plot removing outliers*
  
</div>

We can see the mean price of each categorie with the next boxplots:

![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price%205.png)
</div>
  
<div align="center">

*Prices of each category*
  
</div>


## Insert date:

<div align="center">
  
![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/insert_date.png)
</div>
  
<div align="center">

*Distribution of the dataset in the different years*
  
</div>

**I think that the next graphic is very interesting to see the real evolution of these products.**

Since we are talking about small differences in prices (cents), this graph is much clearer to observe the possible increase

<div align="center">

![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/price6.png)
  
</div>

<div align="center">


*Evolution of the price per month for each year*
  
</div>

## Supermarket

<div align="center">

![alt text](https://github.com/enekoegiguren/supermarket/blob/main/images/supermarket.png)
  
</div>

<div align="center">


*Distribution of the data per supermarket*
  
</div>

We are going to separate the dataframe by supermarkets:

```
df_mercadona = df1[df1.supermarket == 'mercadona-es']
df_dia = df1[df1.supermarket == 'dia-es']
df_carrefour = df1[df1.supermarket == 'carrefour-es']
```

