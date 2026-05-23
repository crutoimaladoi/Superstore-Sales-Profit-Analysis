import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('Superstore.csv')

print(df.head(11))
print(df.tail(11))
print(df.shape)
print(df.columns.tolist())


#-----DATA CLEANING

#-----Missing values & Duplicates----
print('Missing values',df.isna().sum())
print('Duplicates',df.duplicated().sum())

#Drop duplicates
df=df.drop_duplicates()

#Standardize column name
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print(df.head())


#------DATA ANALYSIS-----
total_sales=df['sales'].sum()
total_profit=df['profit'].sum()
total_orders=df['order_id'].nunique()
average_discount=df['discount'].mean()


print(f'Total sales: {total_sales:,.2f}')
print(f'Total profit: {total_profit:,.2f}')
print(f'Total orders: {total_orders}')
print(f'Average discount: {average_discount:,.2f}')


sales_by_region=df.groupby('region')['sales'].mean()
sales_by_State=df.groupby('state')['sales'].mean()
sales_by_category=df.groupby('category')['sales'].mean()
profit_by_category=df.groupby('category')['profit'].mean()



#------Time Analysis
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['ship_date'] = pd.to_datetime(df['ship_date'], dayfirst=True)


#----TOP customers-----

Top_customers=df.groupby('customer_name')['profit'].sum()
top_five=Top_customers.nlargest(10)

repeat_customers = df['customer_name'].value_counts().idxmax()
visit_count = df['customer_name'].value_counts().max()
print(f'Top Customers:{repeat_customers}  with {visit_count} visits')


#----Best & worst selling products

print(df.columns.tolist())
best_selling = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False)
worst_5 = df.groupby('product_name')['sales'].sum().nsmallest(5)
products_with_losses= df[df['profit'] < 0]
sub_categories = df.groupby('sub-category').agg(
    Total_Sales=('sales', 'sum'),
    Total_Profit=('profit', 'sum')
).reset_index()
print('Best selling Products:',best_selling)
print('Worst selling Products:',worst_5)
print('Products with losses: ',products_with_losses)
print('Sub categories',sub_categories)


#----Profit analysis----

region_product_profit = df.groupby(['region', 'product_name'])['profit'].sum().reset_index()
negative_profit_products=df[df['profit'] < 0]['profit'].sum()
#Discount effect on products
df['discounted_sales'] = df['sales'] * (1 - df['discount'])
correlation = df['discount'].corr(df['profit'])

print('Correlation between discount and profit:',correlation)


import seaborn as sns

plt.style.use('seaborn-v0_8')

# 1 Sales by Region
plt.figure()
df.groupby('region')['sales'].sum().plot(kind='bar')
plt.title('Total Sales by Region')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()


# 2 Profit by Category
plt.figure()
df.groupby('category')['profit'].sum().plot(kind='bar', color='green')
plt.title('Total Profit by Category')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.show()

# 3. Monthly Sales Trend
df['order date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')

monthly_sales = df.groupby('month')['sales'].sum()

plt.figure()
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# 4. Discount vs Profit
plt.figure()
plt.scatter(df['discount'], df['profit'], alpha=0.5)
plt.title('discount vs profit')
plt.xlabel('discount')
plt.ylabel('profit')
plt.show()

# 5. Top 10 Customers by Sales
top_customers = df.groupby('customer_name')['sales'].sum().nlargest(10)

plt.figure()
top_customers.plot(kind='bar')
plt.title('Top 10 Customers by Sales')
plt.ylabel('sales')
plt.xticks(rotation=45)
plt.show()

# 6. Sub-Category Profit
subcat_profit = df.groupby('sub-category')['profit'].sum().sort_values()

plt.figure()
subcat_profit.plot(kind='barh')
plt.title('Profit by Sub-Category')
plt.xlabel('Profit')
plt.show()




# 7. Correlation Heatmap
plt.figure()
sns.heatmap(df[['sales', 'profit', 'discount']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



