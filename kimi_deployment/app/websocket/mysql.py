#!/usr/bin/env python
#!-*- coding:utf8 -*-
#!vim: set ts=4 sw=4 sts=4 tw=100 noet:
# ***************************************************************************
#
# Copyright (c) 2016 Orthm.com, Inc. All Rights Reserved
#
# **************************************************************************/
import pymysql
import configparser
Version='V1'
class MySQL():

    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read( "./app/websocket/conf/config.ini")
        mq_host = cf.get("mysql", "host")
        mq_port = cf.get("mysql", "port")
        mq_user = cf.get("mysql", "user")
        mq_passwd = cf.get("mysql", "passwd")
        mq_db = cf.get("mysql", "db")
        try:
            db = pymysql.connect(host=mq_host, port=int(mq_port), user=mq_user, passwd=mq_passwd, db=mq_db,
                             charset='utf8')
        except :
            db = pymysql.connect(host='172.25.81.231', port=3306, user='root', passwd='rootpassword', db='cdss_cihai', charset='utf8')
        self.conn = db

    def reconnect(self):

        try:
            self.conn.ping(reconnect=True)
        except:
            self.conn.reconnect()

    def commit(self):
        self.reconnect()
        self.conn.commit()

    def execute_once(self, query, params):
        self.reconnect()
        cur = self.conn.cursor(buffered=True)
        cur.execute(query, params)
        row_count = cur.rowcount
        cur.nextset()
        cur.close()
        return row_count

    def insert_and_get_id(self, query, params):
        self.reconnect()
        cur = self.conn.cursor(buffered=True)
        cur.execute(query, params)
        last_id = cur.lastrowid
        cur.nextset()
        cur.close()
        return last_id

    def fetch_one(self, query, params):
        self.reconnect()
        cur = self.conn.cursor(dictionary=True, buffered=True)
        cur.execute(query, params)
        result = cur.fetchone()
        cur.nextset()
        cur.close()
        return result

    def fetch_all(self, query, params):
        self.reconnect()
        cur = self.conn.cursor(dictionary=True, buffered=True)
        cur.execute(query, params)
        rlt = list(cur.fetchall())
        cur.nextset()
        cur.close()
        return rlt

    def insert_diagnosis(self, id, name, diagnos, other, other2):
        self.reconnect()
        cur = self.conn.cursor()


        try:
            cur.execute('sql')
            self.conn.commit()
        except:
            ret = -1
            self.conn.rollback()
        cur.close()
        return ret

    def insert_disease(self, id, name, type, value, drug=''):
        self.reconnect()
        cur = self.conn.cursor()
        name = name.replace("'", "\\\'")

        name = name.replace('"', '\\\"')

        try:
            cur.execute('sql')
            self.conn.commit()
        except BaseException as e:
            ret = -1
            print(e)
            self.conn.rollback()
        cur.close()
        return ret

    def connClose(self):
        self.conn.close()

    def get_userdict(self, user_name):
        self.reconnect()
        cur = self.conn.cursor()
        #query = "select * from user where user_name='"+user_name+"'"
        query = "select u.*, g.group_permission, g.group_status from kuser u, kgroup g where u.user_name = '" \
                + user_name + "' and u.group_id = g.group_id"
        #print(query)
        cur.execute(query)
        result = cur.fetchone()
        cur.nextset()
        cur.close()
        if result is None:
            return result
        else:
            print(result)
            user_dict = dict()
            user_dict['name'] = result[0]
            user_dict['status'] = result[6] & result[9]
            user_dict['pwd'] = result[3]
            user_dict['createdate'] = result[4]
            user_dict['endtime'] = result[5]
            user_dict['perm'] = result[8]
            return user_dict
