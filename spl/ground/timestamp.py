'''
HOW TO USE IT::

    from spl.ground.timestamp import Timestamp
    now = Timestamp.now()
    today = Timestamp.today()
    or
    ts = Timestamp()
    now = ts.asNowString()
    today = ts.asTodayString()
'''
import datetime


class Timestamp():
    ''' Class for tracking numbers generation
    '''

    def __init__(self):
        self._now = datetime.datetime.now()

    def asNowString(self):
        '''
        Returns
        -------
        now: string
            yearMonthDay_hoursMinutesSeconds
        '''
        return self._now.strftime("%Y%m%d_%H%M%S")

    def asTodayString(self):
        '''
        Returns
        -------
        today: string
            year month day
        '''
        return self._now.strftime("%Y%m%d")

    @staticmethod
    def now():
        '''
        Returns
        -------
        now: string
            yearMonthDay_hoursMinutesSeconds
        '''
        return Timestamp().asNowString()

    @staticmethod
    def today():
        '''
        Returns
        -------
        today: string
            year month day
        '''
        return Timestamp().asTodayString()

    def __str__(self):
        return self.asNowString()
