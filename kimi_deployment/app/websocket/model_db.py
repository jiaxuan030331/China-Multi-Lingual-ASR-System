import time
from app.websocket.mysql import MySQL


def compare_time(time1, time2):
    s_time = time.mktime(time.strptime(time1, '%Y-%m-%d %H:%M:%S'))
    e_time = time.mktime(time2.timetuple())  # time.strptime(time2, '%Y-%m-%d %H:%M:%S')
    # print('s_time is:', s_time)
    # print('e_time is:', e_time)
    return int(e_time) - int(s_time)
def is_expire(end_time):
    cur_time = int(time.time())
    e_time = int(time.mktime(end_time.timetuple()))
    if cur_time - e_time > 0:
        return True
    else:
        return False

class ModelService:
    def __init__(self):
        print('')
        self.db = MySQL()
    def check_token(self, token):
        self.db.reconnect()
        try:
            cur = self.db.conn.cursor()
            sql = "SELECT * FROM user_type WHERE token='%s'" % (token)
            cur.execute(sql)
            result = cur.fetchone()
            cur.nextset()
            cur.close()
            if result is None:
                return -1, None
            else:
                print(result)
                model_type = result[1]
                user_id = result[2]
                used_count = result[4]
                count = result[5]
                end_time = result[6]
                #cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if is_expire(end_time) is True:
                    return -2, None
                if int(count) < int(used_count):
                    return -3, None
                user_type = dict()
                user_type['type_id'] = result[0]
                user_type['type_name'] = model_type
                user_type['user_id'] = user_id
                user_type['token'] = token
                user_type['used_count'] = used_count
                return 0, user_type


        except Exception as e:
            print(e)
            return -1, None
        return 0, None
    def add_result(self, result_str, sen_id, user_type):
        self.db.reconnect()
        cur = self.db.conn.cursor()
        update_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        sql = "INSERT INTO user_usage(user_id, type_id, token, update_time, key_id, content) " \
              "VALUES (%s, %s, '%s', '%s', '%s', '%s')" % (user_type['user_id'], user_type['type_id'],
                                                           user_type['token'], update_time, sen_id, result_str)

        ret = 0
        try:
            cur.execute(sql)
            self.db.conn.commit()
        except BaseException as e:
            ret = -1
            print(e)
            self.db.conn.rollback()
        cur.close()
        return ret


if __name__ == '__main__':
    mdb = ModelService()
    ret, result = mdb.check_token('S123P38ASD33U7W3')
    print(result)