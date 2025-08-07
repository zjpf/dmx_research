import os


def get_poi(ss, pday):
    def parse_addr(iter0):
        os.environ['MODELSCOPE_CACHE'] = 'pyenv/data'
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        task = Tasks.token_classification
        model = 'pyenv/data/mgeo_geographic_elements_tagging_chinese_base'
        pipeline_ins = pipeline(task=task, model=model)

        data, ret = [], []
        for row in iter0:
            data.append(row['address'])
            ret.append([row['rowkey'], row['address_get_time']])
        ps = pipeline_ins(input=data)
        for i, s in enumerate(ps):
            type_span = {d['type']: d['span'] for d in s['output']}
            ads = [type_span.get(fn, None) for fn in ['prov', 'city', 'district', 'town', 'community', 'road', 'poi', 'subpoi', 'houseno']]
            ret[i].extend(ads)
        return ret

    c_sql = """
    create table if not exists dp_data_db.dm_rowkey_address_parse_poi (
        rowkey string, address_get_time string, prov string, city string, district string, town string, community string,
        road string, poi string, subpoi string, houseno string
    ) partitioned by(pday string) stored as orc
    """
    ss.sql(c_sql)
    df = ss.sql("""
    select rowkey, address, address_get_time
    from fin_dm_data_ai.dm_crs_fu_zhou_text_location_v1_dd
    where length(address)>6 and pday={pday} and address_src='residence' limit 300
    """.format(pday=pday))
    rdd1 = df.rdd.mapPartitions(parse_addr)
    schema = StructType([StructField(fn, StringType(), True) for fn in ['rowkey', 'address_get_time', 'prov', 'city', 'district', 'town', 'community', 'road', 'poi', 'subpoi', 'houseno']])
    predict_result = ss.createDataFrame(rdd1, schema=schema)
    # predict_result.show(n=200, truncate=False)
    predict_result.createOrReplaceTempView("parser_poi")
    sql2 = ("""INSERT overwrite TABLE dp_data_db.dm_rowkey_address_parse_poi partition(pday='{pday}')
                SELECT * FROM parser_poi""".format(pday=pday))
    ss.sql(sql2)


def parse_file(fname, fout):
    import csv
    import pdb
    import time
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    task = Tasks.token_classification
    model = 'iic/mgeo_geographic_elements_tagging_chinese_base'
    pipeline_ins = pipeline(task=task, model=model)
    i, pts = 0, time.time()
    with open(fout, 'w') as fw:
        with open(fname) as fr:
            for s in csv.reader(fr):
                s = s[0]
                ps = pipeline_ins(input=s)
                #pdb.set_trace()
                type_span = {d['type']: d['span'] for d in ps['output']}
                ads = [type_span.get(fn, '') for fn in ['prov', 'city', 'district', 'town', 'community', 'road', 'poi', 'subpoi', 'houseno']]
                fw.write('\"{}\",{}\n'.format(s, ','.join(ads)))
                i += 1
                if i % 50000 == 0:
                    cts = time.time()
                    print("use_time", cts-pts)
                    pts = cts
                    #pdb.set_trace() 

if __name__ == "__main__":
    #from pyspark.sql import SparkSession
    #from pyspark.sql.types import ArrayType, StringType, StructType, StructField
    #ss = SparkSession.builder.enableHiveSupport().getOrCreate()
    #get_poi(ss, pday=20241130)
    import sys
    parse_file(sys.argv[1], fout=sys.argv[2])
