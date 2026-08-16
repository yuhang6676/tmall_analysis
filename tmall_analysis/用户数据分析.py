
import pandas as pd
from pyecharts.charts import Bar,Map3D,Line
import matplotlib.pyplot as plt
import warnings
from pyecharts import options as opts
from pyecharts.globals import ChartType
from pyecharts.commons.utils import JsCode

# 字体设置
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family = 'Microsoft YaHei',size = '15')
warnings.filterwarnings("ignore")


#查看前十行数据
df = pd.read_csv(r"E:\tcl\tmall_order_report.csv")
df.head(10)


# 去除字段名中的空格
new_columns = [col.strip() for col in df.columns]
df.columns = new_columns
# 显示 dataframe 信息
df.info()


# 数据基本描述
print('数据的时间区间为',df['订单创建时间'].min(),'到',df['订单创建时间'].max())
print('收货地址总计有：',df['收货地址'].nunique(),'个')
df.describe()


# 提取日期中的时间为后续分析做准备
df['订单创建时间'] = pd.to_datetime(df['订单创建时间'])
df['订单付款时间'] = pd.to_datetime(df['订单付款时间'])
df['月'] = df['订单付款时间'].dt.month
df['日'] = df['订单付款时间'].dt.day
df2 = df[~df['订单付款时间'].isnull()].copy()
df2['月'] = df2['月'].apply(lambda x:int(x)).astype('str')
df2['日'] = df2['日'].apply(lambda x:int(x)).astype('str')
df2['日期'] = df2['月'] + '月' + df2['日'] + '日'
df2['周'] = df2['订单付款时间'].dt.weekday + 1
df2['周'] = '星期' + df2['周'].astype('str')
df2['月'] = df2['月'].astype('int')
df2['日'] = df2['日'].astype('int')
df2 = df2.sort_values(by = '订单付款时间')
df2['小时'] = df2['订单付款时间'].dt.hour
print(df2.head())


# 查看收货地址信息
print(df2.收货地址.unique())


#优化收货地址信息
df2['收货地址'] = df2.收货地址.apply(lambda x:x.strip('省|自治区'))
df2['收货地址'] = df2.收货地址.replace(['新疆维吾尔','广西壮族','宁夏回族'],['新疆','广西','宁夏'])
df2.head()
print(df2.收货地址.unique())


# 查看缺失数据
print(df[df['订单付款时间'].isnull()].head())


# 查看是否有重复值
df[df['退款金额'] > df['总金额']]
print('重复值数量为：',df.duplicated().sum())
def kde_plot_array(df):
    """
    绘制概率密度图矩阵函数
    df:要绘制图像的dataframe
    绘制各个字段的概率密度分布，最终返回图像的show()
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize = (24,16))   # 设置画布大小
    col_count = len(df.columns)
    row_num = int(round(col_count / 2, 0))

    for num, index in enumerate(df.columns):
        plt.subplot(row_num, 2, num+1)
        sns.kdeplot(df[index], shade = True, label = index, alpha = 0.7)
        plt.legend()
        plt.title(f'{index}分布图', fontsize=14)
        plt.xlabel('')   # 去掉x轴标签，减少文字挤压
        plt.ylabel('Density', fontsize=12)

    plt.subplots_adjust(hspace=0.4)  # 增加子图垂直间距，解决标题文字重叠
    plt.tight_layout()
    return plt.show()

# 过滤极端数据
df.describe()
df[df.总金额 > 5000]
plot_df = df[(df.总金额 < 500)&(df.退款金额 < 400)][['总金额','买家实际支付金额','退款金额']]
kde_plot_array(plot_df)


"""
成交金额在时间维度上的变化
"""

# 1. 数据聚合：按日期统计每日成交总额（和截图里的逻辑完全对应）
change = df2[['买家实际支付金额', '日']].groupby('日').sum().round(2).reset_index().sort_values(by='日')

# 2. 绘图函数：白底、红线、显示圆点和数值
def echarts_line(x, y, title='主标题', subtitle='副标题', label='图例'):
    line = Line(
        init_opts=opts.InitOpts(bg_color='#ffffff')  # 白色背景
    )
    line.add_xaxis(x)
    line.add_yaxis(
        series_name=label,
        y_axis=y,
        is_smooth=True,
        is_symbol_show=True,  # 显示数据圆点
        label_opts=opts.LabelOpts(is_show=True),  # 圆点上显示数值
        linestyle_opts=opts.LineStyleOpts(color='#ff0000', width=3),
        areastyle_opts=opts.AreaStyleOpts(
            color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0, color: 'rgba(255, 0, 0, 0.4)'
            }, {
                offset: 1, color: 'rgba(255, 0, 0, 0.05)'
            }], false)""")
        )
    )

    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            subtitle=subtitle,
            pos_left='center',
            title_textstyle_opts=dict(color='#000000')
        ),
        legend_opts=opts.LegendOpts(is_show=True, pos_left='right', pos_top='3%')
    )
    line.render("每日成交额.html")
    return line


# 3. 调用函数出图

echarts_line(
    x=change['日'].tolist(),
    y=change['买家实际支付金额'].tolist(),
    title='成交金额变化图',
    subtitle='成交金额在时间维度上的变化',
    label='成交金额'
)



# 1. 数据聚合：按周统计每周成交总额）

week_change = df2[['周', '买家实际支付金额']].groupby('周').sum().round(2).reset_index()

# 2. 柱状图函数

def echarts_bar(x, y, title='主标题', subtitle='副标题', label='图例'):
    # 创建柱状图对象，白色背景
    bar = Bar(init_opts=opts.InitOpts(bg_color='#ffffff'))

    # 传入x轴、y轴数据
    bar.add_xaxis(x)
    bar.add_yaxis(
        series_name=label,
        y_axis=y,
        category_gap="50%",  # 柱子之间的间距
        label_opts=opts.LabelOpts(is_show=True)  # 柱子顶部显示数值
    )

    # 配置柱子样式 + 标记点
    bar.set_series_opts(
        # 标记点：最大值、最小值、平均值
        markpoint_opts=opts.MarkPointOpts(
            data=[
                opts.MarkPointItem(type_="min", name="最小值"),
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="average", name="平均值")
            ]
        ),
        itemstyle_opts=opts.ItemStyleOpts(
            color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0, color: 'rgba(255, 0, 0, 0.9)'
            }, {
                offset: 1, color: 'rgba(255, 0, 0, 0.3)'
            }], false)"""),
            border_radius=[10, 10, 0, 0]
        )
    )
    # 全局配置：标题、图例
    bar.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            subtitle=subtitle,
            pos_left='center',
            title_textstyle_opts=dict(color='#000000')  # 标题改黑色，适配白底
        ),
        legend_opts=opts.LegendOpts(
            is_show=True,
            pos_left='right',
            pos_top='3%'
        )
    )
    bar.render(f"{title}.html")
    return bar

# 3. 调用函数生成图表

echarts_bar(
    x=week_change['周'].tolist(),
    y=week_change['买家实际支付金额'].tolist(),
    title='订单成交金额平均每周对比',
    subtitle='每周对比图',
    label='成交金额'
)


"""
时间金额在地区维度上的分布
"""

# 1. 按省份统计成交总额
change_map = df2[['收货地址','买家实际支付金额']].groupby('收货地址').sum().round(2).reset_index().sort_values(by='买家实际支付金额', ascending=False)

# 2. 3D地图函数（低灵敏度 方便微调）
def map3d_with_bar3d(province, data_list, title, label):
    pos = {
        '黑龙江':[127.97,45.37],'上海':[121.46,31.29],'内蒙古':[110.35,41.49],'吉林':[125.82,44.26],
        '辽宁':[123.12,42.12],'河北':[114.50,38.10],'天津':[117.42,39.42],'山西':[112.34,37.94],
        '陕西':[109.12,34.20],'甘肃':[103.59,36.30],'宁夏':[106.36,38.18],'青海':[101.40,36.82],
        '新疆':[87.92,43.59],'西藏':[91.11,29.97],'四川':[103.95,30.76],'重庆':[108.38,30.44],
        '山东':[117.16,36.87],'河南':[113.47,34.62],'江苏':[118.81,31.92],'安徽':[117.29,32.06],
        '湖北':[114.39,30.66],'浙江':[119.53,29.88],'福建':[119.45,25.92],'江西':[116.00,28.66],
        '湖南':[113.08,28.26],'贵州':[106.70,26.77],'广西':[108.48,23.12],'海南':[110.39,19.85],
        '广东':[113.28,23.13],'北京':[116.41,39.90],'云南':[102.71,25.04],'香港':[114.17,22.28],
        '澳门':[113.55,22.20],'台湾':[121.52,25.03]
    }
    for p, v in zip(province, data_list):
        if p in pos:
            pos[p].append(v)
    data = list(zip(pos.keys(), pos.values()))

    map_3d = Map3D(init_opts=opts.InitOpts(bg_color='#ffffff', width='1200px', height='900px'))

    map_3d.add_schema(
        maptype="china",
        itemstyle_opts=opts.ItemStyleOpts(color="#e5e5e5", border_color="#999"),
        map3d_label=opts.Map3DLabelOpts(is_show=False),
        emphasis_label_opts=opts.LabelOpts(is_show=False),
        # 核心修改：大幅降低灵敏度，方便微调
        view_control_opts=opts.Map3DViewControlOpts(
            alpha=60,
            beta=0,
            distance=140,
            rotate_sensitivity=0.2,  # 旋转灵敏度拉低，拖动转动很慢
            zoom_sensitivity=0.5,    # 缩放也放慢
            pan_sensitivity=0.3      # 平移灵敏度也降低，防止拖飞
        )
    )

    map_3d.add(
        series_name=label,
        data_pair=data,
        type_=ChartType.BAR3D,
        bar_size=1.2,
        itemstyle_opts=opts.ItemStyleOpts(color="#ff0000"),
        label_opts=opts.LabelOpts(
            is_show=True,
            color="#000000",
            font_size=12,
            position="top",
            background_color="rgba(255, 255, 255, 0.9)",
            border_color="#cccccc",
            border_width=1,
            padding=[3, 6],
            formatter=JsCode("function(data){return data.name + ' ' + data.value[2];}")
        )
    )

    map_3d.set_global_opts(
        title_opts=opts.TitleOpts(title=title, pos_left='center', pos_top='10px'),
        legend_opts=opts.LegendOpts(pos_left='right', pos_top='3%')
    )

    map_3d.render(f"{title}.html")
    print("生成完成：全国成交金额分布图.html")

# 3. 调用生成
map3d_with_bar3d(
    province=change_map['收货地址'].tolist(),
    data_list=change_map['买家实际支付金额'].tolist(),
    title='成交金额分布图',
    label='成交金额'
)


"""
退款金额在时间维度上的分布
"""

back_money = df2[['日', '退款金额']].groupby('日').sum().round(2).reset_index()
echarts_bar(
    x=back_money['日'].tolist(),
    y=back_money['退款金额'].tolist(),
    title='退款金额日变化图',
    subtitle='每日退款金额',
    label='退款金额'
)


"""
退款金额在地区维度上的分布
"""

local_back_money = df2[['收货地址','退款金额']].groupby('收货地址').sum().round(2).reset_index().sort_values(by='退款金额', ascending=False)
map3d_with_bar3d(
    province=local_back_money['收货地址'].tolist(),
    data_list=local_back_money['退款金额'].tolist(),
    title='退款金额分布图',
    label='退款金额'
)

