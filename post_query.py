#!/usr/bin/env python
# -*- coding: utf-8 -*-
# module providing psql SELECT-like behaviour on lists of tuples such as those
# returned by pgqueryobject.getresult(). Use this module 
# to facilitate re-organization, aggregation and tabulation of existing 
# query results to achieve the desired presentation without 
# having to introduce complex changes to the actual psql queries.
# This will keep the queries general enough to be reused in child reports
# leaving them room to re-organize and aggregate query results as per their 
#     own requirements
# 
# Recommended Usage: 
#    1- Create a post_query.TableSet instance
#    2- Call post_query.post_sql to carry out aggregation and/or joining
#    3- Call TableSet.tabulate() or one of its variants to generate 
#        a tabular representation of the data.

from itertools import groupby
from functools import partial
from decimal import Decimal
import operator
from copy import deepcopy #,copy
from textwrap import dedent
from _collections import defaultdict
from operator import methodcaller
from optparse import OptionParser
from itertools import izip_longest
from string import Formatter
from types import NoneType

itemgetter = operator.itemgetter
try:
    import sys; sys.path.append('/home/nkhoury/bin/pysrc')
    import pydevd; pydevd.settrace('192.168.7.60', suspend=False)
except:
    pass

# import types, dis # useful for debugging compiled code objects to be eval'ed 

#------------------ Data type formatting and special handling
def identity(val): return val # function returning same element as one passed
def absorbing(val, default=None): return default
def gi(intval, default=0): return int('{0}'.format(intval or default))
def gd2(floatval, precision=2, default=Decimal('0.0')): 
    return Decimal('{0:.{1}f}'.format(floatval or default,precision))
def gs(strval, default=''): return '{0}'.format(strval or default)
def gis(val, strdefault='',intdefault=0):
    if isinstance(val, (str,NoneType)):
        return gs(val, default=strdefault)
    else:
        return gi(val, default=intdefault)
def gd2s(val, strdefault='',gd2default=Decimal('0.0')):
    if isinstance(val, (str,NoneType)):
        return gs(val, default=strdefault)
    else:
        return gd2(val, default=gd2default)
format_ranks = [identity, gi, gs, gd2]
def bestformat(*args):
    _maxrank= max([format_ranks.index(a) for a in args])
    return format_ranks[_maxrank]
def counterlabel(*args):
    '''Ignores arguments, increments then returns number of calls 
            since last counterlabel.counter = 0'''
    counterlabel.counter += 1
    return counterlabel.counter
counterlabel.counter = 0
def emptylabel(*args): 
    '''ignores arguments and returns empty string'''
    return '' 
def fixedlabel(label, *args):
    '''Always returns label''' 
    return lambda *x: label
def defaulter(func, default):
    '''
    wrapper for func that returns a default value 
        when func return evaluates to False
    ''' 
    def wrapper(*args,**kwargs):
        return func(*args, **kwargs) or default
    return wrapper

def Struct(*elements,**kwargs):
    def init(self,**kwargs):
        for kw,kv in kwargs.items():
            self.__setattr__(kw,kv)
    def update(self, **kwargs):
        [setattr(self,kw,kwargs[kw]) for kw in kwargs if kw in self.__slots__]
    def todict(self): 
        return dict([(s,getattr(self,s)) for s in self.__slots__])
    elements = elements+tuple(kwargs.keys())
    struct_cls = type('', (object, ), dict(__slots__=elements,
                                           __init__=init,
                                           update=update,
                                           todict=todict))
    return struct_cls(**kwargs)
    
# def safeitemgetter(arg): 
#    return itemgetter(*arg) if hasattr(arg,'__iter__') else itemgetter(arg)
def itemsetter(*items):
    ''' Similar to operator.itemgetter but sets the value of the provided
            keys/indices instead of getting them'''
    if not items:
        raise ValueError('Must pass at least one index')
    elif len(items) == 1:
        item = items[0]
        def setter(obj, value):
            obj[item] = value
    else:
        def setter(obj, *values):
            if len(values) < len(items): 
                values = values[0] # assume caller passed a list explicitly
            for item, value in zip(items, values):
                obj[item] = value
    return setter

#---------------- TESTS
burgers_and_fajitas =  [('76994',[(0, 'burgers', 0.0), 
                                  (1, 'burgers', 1.1), 
                                  (0, 'drinks', 2.2), 
                                  (1, 'drinks', 3.3), 
                                  (1, 'steaks', 4.4)]), 
                        ('90219',[(0, 'burgers', 0.0), 
                                  (0, 'drinks', 2.3), 
                                  (1, 'burgers', 1.2), 
                                  (1, 'drinks', 3.4), 
                                  (1, 'steaks', 4.5)]), 
                        ('75495',[(0, 'burgers', 0.0), 
                                  (0, 'fajitas', 3.2), 
                                  (1, 'Fajitas', 4.3), 
                                  (1, 'burgers', 2.1), 
                                  (1, 'salads', 5.4)])]

class TableWriterErr(ValueError):pass

class TableWriter(object):
    hide_total_column = False
    detailed = True
    entities = ['76994', '90219', '75495']
    TAGS = HEADERS = CONDENSE = DATA = ''
    FS = '|' # field separator
    TS = '|' # entity seprator
    FSS = len(FS) # field separator span
    TSS = len(TS) #entity separator span
    
    def __init__(self):
        self.row_open = False
        self.current_row = []
        self.open_entity_header_row = self.open_row
        self.open_field_header_row = self.open_row
        self.open_summary_row = self.open_row
        self.total_row = self.open_row

    def open_row(self, formatting=''):
        if self.row_open: raise TableWriterErr("Row already open")
        self.row_open = True
        self.current_row = ['']
                            
    def close_row(self):
        if not self.row_open: raise TableWriterErr("No Open Row")
        print self.FS.join(self.current_row) + self.TS
        self.row_open = False
        self.current_row = []
        
    def add_cell(self, text, colspan=1):
        if not self.row_open: raise TableWriterErr("No Open Row")
        self.current_row.append('{0!s:{1}.{1}}'.format(text, colspan))

def test_value_set():
    grss=ValueSet('grss', gd2,
                  [('76994',1.23),
                   ('90219',32.34),
                   ('aroma bar', 3.43)])
    tax=ValueSet('tax', gd2,
                 [('76994',.2),
                  ('90219',3.12),
                  ('aroma bar',0.4)])
    net = grss - tax
    net2 = ValueSet('net',net)
    print '{grss!r}'.format(grss=grss)
    print '{tax!r}'.format(tax=tax)
    print '{net!r}'.format(net=net)
    print '{net2!r}'.format(net2=net2)
    print 'ValueSet => TableSet:'
    gross_tax_net = TableSet(grss, tax, net2)
    gross_tax_net.condense([('CATEGORIES',fixedlabel('[TOTAL]'), 15), 
                            ('TAX','tax', 5), 
                            ('GROSS SALES','grss',10), 
                            ('NET SALES','net',10)],
                           table_writer=TableWriter(), 
                           entity_headers=True,field_headers=True,
                           summary_column=True, summary_row=True)
    tax2=ValueSet('tax', gd2,
                  [('7***4',.2),('90219',3.12),('aroma bar',0.4)])
    try:
        net = grss - tax2
    except ValueSetErr, err:
        print err

def test_total_sets():
    _tableset = TableSet([('oprid', gi),('catnm', gs),('grss', gd2)], 
                         tables=burgers_and_fajitas)
    print _tableset
    print """--"""
    print """TableSet.total_sets('grss')"""
    print """==>"""
    print """{0!r}""".format(_tableset.total_sets('grss'))

def test_tabulate():
    _tableset = TableSet('oprid, catnm, grss',
                          gi   , gs   , gd2  ,
                          tables=burgers_and_fajitas)
    print _tableset 
    _tableset.defineformulae([('net',gd2,'{grss} / gd2(1.13)')])
    # print the unbalanced record vectors
    for _rrv in _tableset.record_vectors(['catnm','oprid','grss']):
        print '||'.join(['{0:10.10}|{1!s:3.3}|{2!s:5.5}'.format(*f) 
                         for f in _rrv])
    # print balanced tabulation 
    _tableset.tabulate([('CATEGORIES','catnm', 15), 
                        ('OPERATORS','oprid', 5), 
                        ('GROSS SALES','grss',10), 
                        ('NET SALES','net',10)],
                       table_writer=TableWriter(), 
                       entity_headers=True, 
                       field_headers=True, 
                       summary_column=True, 
                       summary_row=True)

def test_totals():
    _tableset = TableSet([('oprid', gi), 
                          ('catnm', gs), 
                          ('grss', gd2)], 
                         tables=burgers_and_fajitas)
    print _tableset
    print """--\nTableSet.totals('grss', '76994')"""
    print """==>\n{0}""".format(_tableset.totals('grss', '76994'))
    print """--\nTableSet.totals('grss')"""
    print """==>\n{0}""".format(_tableset.totals('grss'))
    print """--\nTableSet.totals(['oprid','grss'])\n"""
    print """==>\n{0}""".format(_tableset.totals(['oprid','grss']))
    # invalid total field
    try:
        print """_tableset.totals('gross')"""
        _tableset.totals('gross')
    except TableModelErr, err:
        print 'Exception Text: ', err
        print 'Exception fields', err.fields
        print 'Exception model', err.model
        
    #invalid table set tuple:
    try:
        TableSet('oprid', tables=burgers_and_fajitas)
    except TableSetErr, err:
        print err

def test_aggregate():
    from_tableset = TableSet([('oprid', gi), 
                              ('catnm', gs), 
                              ('grss', gd2)], 
                             tables=burgers_and_fajitas)
    print from_tableset
    
    print """
    >aggregate_table = post_sql(SELECT='catnm,grss,oprid', 
                                SUM=        'grss',
                                FROM=from_tableset,
                                WHERE= "{catnm} <> 'drinks'",
                                GROUPBY=         'oprid', 
                                ORDERBY="{oprid},{catnm}.lower()")"""
    aggregate_table = post_sql(SELECT='catnm,grss,oprid', 
                               SUM=        'grss',
                               FROM=from_tableset,
                               WHERE= "{catnm} <> 'drinks'",
                               GROUPBY=         'oprid', 
                               ORDERBY="{oprid},{catnm}.lower()")
    print aggregate_table
    try:
        print "aggregate_table = post_sql(SELECT='gross')"
        aggregate_table = post_sql(SELECT='gross',
                                   FROM=from_tableset)
    except TableModelErr, e:
        print e
        print e.fields
        print e.model

def test_joins():
    ''' test left and inner joins '''
    def str_table(table):
        _str = ''
        for r in table:
            _str += '\t'.join(map(repr,r)) + '\n'
        return _str
        
    print """
    >outlet_list=['76994',
                  '90219',
                  '75495']
    >left_model = TableModel(['oprid',
                              'catnm',
                              ('grss',gd2)])"""
    outlet_list=['76994',
                 '90219',
                 '75495']
    left_model = TableModel(['oprid',
                             'catnm',
                             ('grss',gd2)])
    print left_model
    print """
    >right_model = TableModel(['oprid',
                               ('catnm', gs),
                               ('grss', gd2)])"""
    right_model = TableModel(['oprid',
                              ('catnm', gs),
                              ('grss', gd2)])
    print right_model
    print """-- Left-of-Join table
    >left_table =   [(0, 'burgers', 0.0), 
                     (1, 'burgers', 1.1), 
                     (0, 'drinks', 2.2), 
                     (1, 'drinks', 3.3), 
                     (1, 'steaks', 4.4)]"""
    #                0   1    2
    left_table =   [(0, 'burgers', 0.0), 
                    (1, 'burgers', 1.1), 
                    (0, 'drinks', 2.2), 
                    (1, 'drinks', 3.3), 
                    (1, 'steaks', 4.4)]
    print str_table(left_table)
    print """-- Right-of-Join table
    >right_table =  [(0, 'burgers', 0.1), 
                     (1, 'burgers', 1.2), 
                     (0, 'drinks', 2.3), 
                     (1, 'drinks', 3.4)]"""
    right_table =  [(0, 'burgers', 0.1), 
                    (1, 'burgers', 1.2), 
                    (0, 'drinks', 2.3), 
                    (1, 'drinks', 3.4)]
    print str_table(right_table)
    print """-- JoinModel
    >join_model = JoinModel(left_model,
                            right_model,
                            select=['grss','right.oprid','right.catnm'],
                            using=['catnm'])"""
    join_model = JoinModel(left_model, 
                           right_model, 
                           select=['grss','right.oprid','right.catnm'],
                           using=['catnm'])
    print join_model
    print """-- post_query.join
    >join(left_table,
          right_table, 
          join_model)=>"""
    print str_table(join(left_table,
                         right_table,
                         join_model))
    print """-- post_query.leftjoin
    >leftjoin(left_table,
              right_table,
              join_model)=>"""
    print str_table(leftjoin(left_table,
                             right_table,
                             join_model))
    print """-- Left-of-join TableSet
    >left_set = TableSet(left_model,
                         table0=left_table)
    >print repr(left_set)"""
    left_set = TableSet(left_model,
                        table0=left_table)
    print repr(left_set)
    print """-- Right-of-join TableSet
    >right_set = TableSet(right_model)
    >for outlet in outlet_list: right_set.addtable(outlet, right_table)
    >print repr(right_set)"""
    right_set = TableSet(right_model) 
    for outlet in outlet_list: right_set.addtable(outlet, right_table)
    print repr(right_set)
    print """-- JOIN with TableSet.join
    >joined_set = left_set.join(right_set,
                                select=['grss',
                                        'right.oprid',
                                        'right.catnm'],
                                using=['catnm'])
    >print repr(joined_set)"""
    joined_set = left_set.join(right_set,
                               select=['grss',
                                       'right.oprid',
                                       'right.catnm'],
                               using=['catnm'])
    print repr(joined_set)
    print """-- LEFT JOIN with post_query.post_sql
    >joined_set = post_sql(SELECT=['grss','oprid','right.catnm'],
                           FROM = left_set,
                           LEFTJOIN = right_set, 
                           USING=['oprid','catnm'])
    >print repr(joined_set)"""
    joined_set = post_sql(SELECT=['grss','oprid','right.catnm'],
                          FROM = left_set,
                          LEFTJOIN = right_set, 
                          USING=['oprid','catnm'])
    print repr(joined_set)
    #JoinModelErr
    print """-- JoinModelErr Exception
    >joined_set = post_sql(SELECT=['left.grss','right.grss'],
                           FROM = left_set,
                           LEFTJOIN = right_set,
                           USING=['oprid','catnm'])"""
    try:
        joined_set = post_sql(SELECT=['left.grss','right.grss'],
                              FROM = left_set,
                              LEFTJOIN = right_set,
                              USING=['oprid','catnm'])
    except JoinModelErr, err:
        print 'repr(err): ', repr(err)
        print 'Exception Text: ', str(err)
        print 'Exception fields: ', err.fields
        print 'Exception model: ', err.model

class TableModelErr(ValueError):
    AFIELD_IN0_NOT_IN1 = "One or more fields in {0} are not in model:\n{1}"
    def __init__(self, err=AFIELD_IN0_NOT_IN1,fields='', model=''):
        super(TableModelErr, self).__init__(err.format(fields, model))
        self.fields = fields
        self.model = model
    
class TableModel(object):
    '''
    Class encapsulating field lookups,access and formatting
        on a given TableSet/ValueSet
    @keyword model_tuples: a list of field specs: [(namei,formatteri),]
    @keyword formulae: list of tuples: [(namei,codei),]
    '''
    def __init__(self, model_tuples=[], formulae=[]):
        self.model_tuples = []
        self._fldnames = []
        self._fldindices = {}
        self._fldgetters = {}
        self._fldformatters = {}
        for i,fieldmodel in enumerate(model_tuples):
            if isinstance(fieldmodel, tuple): # tuple/fully qualified
                self.model_tuples.insert(i, (fieldmodel[0],fieldmodel[1]))
            else:# single value
                self.model_tuples.insert(i, (fieldmodel, identity))
            fieldname, fieldformatter = self.model_tuples[i]
            self._fldnames.append(fieldname)
            self._fldgetters[fieldname] = itemgetter(i)
            self._fldformatters[fieldname] = fieldformatter
            self._fldindices[fieldname] = i
        # map formulae names and definitions
        self._formulae = dict()
        if formulae:
            self.defineformulae(formulae)
    
    # reminiscent of pgqueryobject.listfields,fieldname,fieldnum
    def listfields(self): return self._fldnames 
    def fieldname(self, i): 
        try: return self._fldnames[i]
        except: raise TableModelErr(fields=i,model=self)
    def fieldnum(self, name): 
        try: return self._fldindices[name]
        except: raise TableModelErr(fields=name, model=self)
    def fieldformatter(self, name): 
        try: return self._fldformatters[name]
        except: raise TableModelErr(fields=name,model=self)
    def fieldgetter(self, name): 
        try: return self._fldgetters[name]
        except: raise TableModelErr(fields=name,model=self) 
    def formattedgetter(self, name,record):
        try: return self._fldformatters[name](self._fldgetters[name](record))
        except: raise TableModelErr(fields=name,model=self) 
    def callonfields(self, func): 
        ''' 
        Returns dict by field names storing results of an arbitrary
            function call on each field in model
        @param func: must be a callable that takes a kw parameter 'name'
        '''
        return dict([(name, func(name=name)) for name in self._fldnames])
     
    def fieldexists(self, name): 
        return self._fldindices.has_key(name) and True
    
    def fieldnums(self, names):
        if not hasattr(names, '__iter__'): names = [names]
        try: return [self._fldindices[name] for name in names]
        except: raise TableModelErr(fields=names, model=self)
    def fieldformatters(self, names): 
        if not hasattr(names, '__iter__'): names = [names]
        try: return [self._fldformatters[name] for name in names]
        except: raise TableModelErr(fields=names, model=self)
    def fieldsgetter(self, names, tuplize=False): 
        if not hasattr(names, '__iter__'): names = [names]
        try:
            if len(names) == 1 and tuplize:
                # generate getter that return a tuple even with single field
                return lambda record: (record[self._fldindices[names[0]]],)
            else:
                return itemgetter(*[self._fldindices[name] for name in names]) 
            return itemgetter(*[self._fldindices[name] for name in names])
        except: 
            raise TableModelErr(fields=names, model=self)
    def fieldssetter(self, names):
        if not hasattr(names, '__iter__'): names = [names]
        try:return itemsetter(*[self._fldindices[name] for name in names])
        except: raise TableModelErr(fields=names, model=self)
    def columngetter(self, field, table):
        _fldgetter = self.fieldgetter(field)
        _fldformatter = self.fieldformatter(field)
        return [_fldformatter(_fldgetter(r)) for r in table]
    
    def totals(self, fields, table):
        '''Given a field name, finds the total of that field in all records 
            and returns it in field format'''
        def caller(f, v): return f(v)#returns a call on f with v as arg
        if not all(map(self.fieldexists, fields)):
            raise TableModelErr(fields=fields, model=self)
        _totals = []
        for f in fields:#sum values of each column
            ccvvs = self.columngetter(f, table)
            #attempt to sum, if no support return first non-None value
            try:
                cttl = sum(ccvvs)
            except:
                cttl = None
            finally:
                _totals.append(cttl)
        # format totals using respective field formatters
        # otherwise we just return a list of totals
        formatted_totals = map(caller, self.fieldformatters(fields), _totals)
        return formatted_totals
    
    def cleanformulae(self): self._formulae = {}
    def defineformulae(self, formula_tuples):
        '''
        Attaches a formula to the table model. 
        @note: The new 'column' doesn't affect table model 
            or query-like operations and is soley used in @tabulate and @totals
        Usage: 
            tableset.defineformulae([('net','gd2({grss} / gd2(1.13))')])
        '''
        # must be a list of 2-element tuples
        assert(formula_tuples and len(formula_tuples[0]) in (2,3))
        # add/update formula
        for tup in formula_tuples:
            try:
                fname, formatter, formula = tup
            except:#perhaps the formatter is abscent. Use identity
                fname, formula = tup
                formatter = identity
            self._formulae[fname]= dict(F=formatter,D=formula)
    def compileformulae(self, formulae=None, submodel=None):
        '''
        Create equivalent code objects, compile and return them
        '''
        # if no model is provided, assume formulae operate on full records
        if not submodel: submodel = self.model
        if not formulae: formulae = self._formulae.keys()
        compiledformulae = dict.fromkeys(formulae)
        formula_formatters = dict.fromkeys(formulae)
        # we need to save this for formula eval call to work
        self._sub_model = submodel
        call_template = """self._sub_model.formattedgetter('{name}',record)"""
        for fname in formulae:
            _getter_calls = self._sub_model.callonfields(call_template.format)
            _formula_def = self._formulae[fname]['D']
            _formula_def = _formula_def.format(**_getter_calls)
            compiledformulae[fname] = compile(_formula_def, '<string>', 'eval')
            formula_formatters[fname] = self._formulae[fname]['F']
        return compiledformulae, formula_formatters
    
    def resolvereference(self, reference):
        '''
        @param reference: a reference in one of the following forms:
            - individual model field name (str)
            - formula name that references one or more model fields (str)
            - callable that doesn't require arguments
        Return list of fields used in reference or a callable used to
            determine a value required during tabulation
        '''
        try:
            if isinstance(reference,str):
                try:
                    # assume label is a field name
                    self.fieldgetter(reference)
                    referenced_fields = [reference]
                    referenced_call = None
                except TableModelErr:#not a field name, a formula perhaps?
                    assert (reference in self._formulae.keys())
                    formula_def = self._formulae[reference]['D']
                    #determine fields referenced by formula
                    referenced_fields = [s[1] 
                                         for s in Formatter().parse(formula_def) 
                                         if s[1]]
                    referenced_call = None
            else:
                # or a callable
                reference()# test it to see if it actually works
                referenced_call = reference
                referenced_fields = []
        except Exception, err:
            raise ValueError(str(err) + """
            reference must be:
            - valid field name
            - a formula referencing valid field names
            - callable with no mandatory arguments """)
        return referenced_call, referenced_fields
    
    def __repr__(self):
        _str = lambda t: '{0!r}\t{1!r}'.format(*t)
        _repr = 'Column\tType\n'
        _repr += '\n'.join(map(_str,self.model_tuples))
        return _repr
    
    def __str__(self):
        _str = lambda t: '{0!s}\t{1!s}'.format(*t)
        _repr = 'Column\tType\n'
        _repr += '\n'.join(map(_str,self.model_tuples))
        return _repr

    def matchfield(self, name, qualifier='.'):
        '''
        Return field name in the form (qualified_name, unqualified_name)
            unqualified name == name
            qualified name can be .name, left.name, right.name, etc...
        raises TableModelErr if field name not in model
        '''
        if self.fieldexists(name): 
            return qualifier+name,name
        else:# try without prefix
            unqualified = name[name.rfind(qualifier)+len(qualifier):]
            if self.fieldexists(unqualified):
                return name, unqualified
            else: 
                raise TableModelErr(fields=name, model=self)

    def submodel(self, fields, tuples=False):
        '''Returns a new model based on selected fields in this model'''
        _sub_model_tuples = []
        for field in fields:
            _sub_model_tuples.append(tuple((field, self.fieldformatter(field))))
        if tuples:
            return _sub_model_tuples
        else:
            return TableModel(_sub_model_tuples)
        
    def __eq__(self, tablemodel):
        '''
        Return whether this TableModel is equivalent to an existing TableModel
            Two TableModels are equivalent if they have the same fields
            and formatters in the same order 
        @attention: This could return false negatives:
            If for a given field, the two TableModels use formatters 
            that are defined in two different modules but have same code,
                self == tablemodel will return false.
            The only reasonable way to deal with such scenario 
            would be to compare the codes of the two formatters:
                ff1 = self._fldformatters[i]
                ff2 = tablemodel._fldformatters[i]
                ff1.__code__.co_code == ff2.__code__.co_code
            which may be a significant overhead 
        '''
        same_fields = self._fldnames == tablemodel._fldnames
        same_formatters = self._fldformatters == tablemodel._fldformatters
        return same_fields and same_formatters
    

class AggModel(TableModel):
    '''
    Encapsulates data and operations needed to access and aggregate a given
        table-set, whose model is from_model.
    An AggModel instance wraps two TableModel instances: 
        1- the inherited/is-a instance which will serve as the model for 
            the resulting table-set
        2- the member/has-a from_model instance that is used to access records
            of the table being aggregated. 
    '''
    def __init__(self, 
                 from_model,
                 select=[], 
                 sums=[], 
                 where='', 
                 group_by=[], #also, callable
                 order_by=''):
        # start by initializing inherited part. this will be the model 
        #   for the resulting aggregated table
        result_model_tuples = AggModel.result_model_tuples(from_model, select)
        super(AggModel, self).__init__(result_model_tuples)
        # keep from-table model to facilitate access to fields in 
        #   from-table via getters/formatters
        self.frommodel = from_model
        self._groupby = group_by
        self._order_by = order_by
        self._sums = sums
        self._where = where
        # sort key to use for groupby call
        self._mk_grpby_sortkey = self.mk_grpby_sortkey(group_by)
        # make _groupby_setter and _selected_groupbys
        self.mk_groupby_setter(group_by, select)
        self._aggregate_record0 = [func(None) 
                                   for func in self.fieldformatters(select)]
        self._groupby_keys = []
        self._where_filter = self.mk_where_filter()
        self._order_key = self.mk_orderby_key()

    def mk_where_filter(self): 
        '''
        Create the code object that needs to be eval'ed 
            when where_filter() is called
        if we have a where clause:
          - generates the itemgetters for all fields in the from-table 
                and places them into a dictionary
          - generates the string of code that, when compiled, will call 
                on the itemgetters and place in dictionary
          - use the second dictionary to replace field entities 
                in the where-clause with respective itemgetter calls
          - compile resulting code and return the code object for repeated use.
        otherwise we return a None-returning function
        '''
        if self._where:
            self._from_getters = self.frommodel._fldgetters
            call_template = """self._from_getters['{name}'](record)""".format
            from_getter_calls = self.frommodel.callonfields(call_template)
            #replace the field entities with the respective getters
            where_filter_code = self._where.format(**from_getter_calls)
            # return code object encapsulating where clause
            return compile(where_filter_code, '<string>', 'eval')
        else:
            return lambda record: None 
    def where(self, records): 
        '''function to return list of records that satisfy where clause'''
        if self._where:
            return filter(self.where_filter, records) # filter list of records 
        else:
            return records
    def where_filter(self, record): 
        '''filter function to determine if a record satisfies where clause'''
        return eval(self._where_filter)

    def mk_grpby_sortkey(self, groupby_flds): 
        '''generate the groupby/sort key function'''
        if groupby_flds:
            if hasattr(groupby_flds, '__call__'):
                groupby_getter = groupby_flds
            else:
                groupby_getter = self.frommodel.fieldsgetter(groupby_flds)
            return lambda record: groupby_getter(record)
        else:
            return lambda record: None

    def mk_orderby_key(self):
        '''
        Generates the code object that needs to be eval'ed when 
            orderkey() is run
        if we have an order-by clause:
            - Generate the getters for all fields in the resulting aggregate 
                table and map by field name
            - Generate code string that, when compiled, 
                will result in calls on the getters. map code by field name
            - Use the code dictionary to replace field entities in the 
                order-by-clause with respective getter calls
            - Compile resulting code and return code object for repeated use.
        otherwise we return a None-returning function
        '''
         
        if self._order_by:
            self._agg_getters = self._fldgetters
            call_template = """self._agg_getters['{name}'](record)""".format
            _agg_getter_calls = self.callonfields(call_template)
            _order_key_code = self._order_by.format(**_agg_getter_calls)
            return compile(_order_key_code, '<string>', 'eval')
        else:
            return lambda record: None
    def order(self, table):# sort aggregated table
        if table and self._order_by:
            list.sort(table, key=self.orderkey)
        return table
    def orderkey(self, record):
        ''' 
        Returns the sort key function to use when sorting/ordering records
        ''' 
        return eval(self._order_key)

    def mk_groupby_setter(self, groupby_flds, selected_flds):
        ''' Initializes self.groupby_setter to be used for padding'''
        # padding will only need to set group-by fields if they're in select.
        selected_groupby_inds = []
        if hasattr(groupby_flds, '__call__'):
            pass
        elif hasattr(groupby_flds, '__iter__'):
            for i, field in enumerate(groupby_flds):
                if field in selected_flds: selected_groupby_inds.append(i)
        else:
            msg='AggModel group_by: Expected list, got: ' + str(groupby_flds)
            raise ValueError(msg)
        if not selected_groupby_inds:
            # no groupbys are selected => pad with nil record
            self._selected_groupbys = identity
            self._groupby_setter = lambda record, groupby_vals: None 
        elif len(selected_groupby_inds) < len(groupby_flds):
            # pad with nil record and set selected group-by fields
            self._selected_groupbys = itemgetter(*selected_groupby_inds)
            self._groupby_setter = \
                self.fieldssetter(self._selected_groupbys(groupby_flds))
        else: #selected_groupby_inds == groupby_flds
            # pad with nil record then set all group-by fields
            self._selected_groupbys = identity
            self._groupby_setter = self.fieldssetter(groupby_flds)

    def pad(self, aggregate_dict):
        '''
        Pad aggregated dict. This can be used to ensure all 
            resulting entity tables are balanced (i.e. in each entity table,  
            a record exists for each group-by value tuple)
        Returned dict maps group-by value tuples to selected/aggregated values
        '''
        for k in self._groupby_keys:
            if not k in aggregate_dict:
                pad_record = deepcopy(self._aggregate_record0)
                self._groupby_setter(pad_record,self._selected_groupbys(k))
                aggregate_dict[k] = tuple(pad_record)
        return aggregate_dict
    
    def aggregate(self, table):
        ''' 
        Carries out all aggregation steps based on model
        Does not do padding or ordering
        ''' 
        table = self.where(table) #apply 'WHERE clause' 
        # group table records based on 'GROUP BY' clause
        _groupby = groupby(sorted(table,key=self._mk_grpby_sortkey),
                           key=self._mk_grpby_sortkey)
        grouped_table = dict([(k,list(g)) for k,g in _groupby])
        aggregate_dict = dict.fromkeys(grouped_table.keys())
        for k in grouped_table:
            if k not in self._groupby_keys: self._groupby_keys.append(k)
            # avoid sharing fields among records ...
            aggregate_record = deepcopy(self._aggregate_record0)
            for field in self.listfields(): # use new columns order 
                i = self.fieldnum(field)
                from_i = self.frommodel.fieldnum(field)
                formatter = self.fieldformatter(field)
                # sum summed field, and format current value for other selects
                if field in self._sums:
                    aggregate_record[i] = sum([formatter(record[from_i]) 
                                               for record in grouped_table[k]])
                else:
                    aggregate_record[i] = formatter(grouped_table[k][0][from_i])
            aggregate_dict[k] = tuple(aggregate_record) 
        return aggregate_dict
           
    @staticmethod 
    def result_model_tuples(table_model, select=[]):
        return table_model.submodel(select, tuples=True)
    
    def __repr__(self):
        where_str = 'WHERE ' + self._where
        if hasattr(self._groupby, '__call__'):
            group_by_str = 'GROUP BY ' + str(self._groupby)
        else:
            group_by_str = 'GROUP BY ' + ','.join(self._groupby)
        order_by_str = 'ORDER BY ' + self._order_by
        _repr = {'select': 'SELECT ',
                 'where': self._where and where_str or '',
                 'group_by':self._groupby and group_by_str or '', 
                 'order_by':self._order_by and order_by_str or ''}
        for i, field in enumerate(self.listfields()):
            if field in self._sums:
                _repr['select'] += ' SUM({field})'.format(field=field)
            else:
                _repr['select'] += '{field}'.format(field=field)
            _repr['select'] += i < len(self.listfields()) - 1 and ', ' or ''
                
        ret =   """
                {select}
                FROM table
                {where}
                {group_by}
                {order_by}
                """.format(**_repr)
        return super(AggModel, self).__repr__() + dedent(ret)

    def __str__(self):
        return self.__repr__()

def aggregate(from_table,aggregatemodel):
    aggregate_dict = aggregatemodel.aggregate(from_table) or {}
    aggregate_table = aggregate_dict.values()
    aggregatemodel.order(aggregate_table)
    return aggregate_table

class JoinModelErr(ValueError):
    DUPLICATEFIELD0_IN1 = 'duplicate field {0} in select {1}'
    def __init__(self, err=DUPLICATEFIELD0_IN1, fields='', model=''):
        super(JoinModelErr, self).__init__(err.format(fields, model))
        self.fields = fields
        self.model = model
    
class JoinModel(TableModel):
    def __init__(self, lmodel, rmodel, select=[], using=[], on=None):
        qual_selects,unqual_selects = [],[]
        res_tuples = JoinModel.result_model_tuples(lmodel,rmodel,
                                                   select,
                                                   qual_selects,unqual_selects)
        super(JoinModel,self).__init__(res_tuples)
        self.__repr = {'selects' : select, 
                       'qual_selects' : qual_selects, 
                       'usings' : using, 'ons' : on} # for repr
        self._concat_model = JoinModel.concat(lmodel,rmodel)
        self._concat_flds_getter = \
            self._concat_model.fieldsgetter(qual_selects)
        self._concat_fld_frmtrs = \
            self._concat_model.fieldformatters(qual_selects)
        self._r_fld_frmtrs = rmodel.fieldformatters(rmodel.listfields())
        self._on = on
        if self.isJoinOn():
            self.join = self.joinon
            self.leftjoin = partial(self.joinon, left_join=True)
        else:
            self.join = self.joinusing
            self.leftjoin = partial(self.joinusing, left_join=True)
            self._l_using_getter = itemgetter(*lmodel.fieldnums(using))
            self._r_using_key = rmodel.fieldsgetter(using)
        self._r_fields0 = tuple(formatter(0) 
                                for formatter in self._r_fld_frmtrs)
    
    def isJoinOn(self): return self._on and True or False
    def isJoinUsing(self): return not self.isJoinOn()
    
    def joinedfields(self, concat_record):
        '''Returns joined fields getter for concatenated records'''
        def formatfields(fields):
            if not hasattr(fields, '__iter__'): fields = [fields] 
            return tuple(formatter(fields[i]) 
                         for i,formatter in enumerate(self._concat_fld_frmtrs))
        return formatfields(self._concat_flds_getter(concat_record))
    

    # WARNING: joinon IS UNTESTED AND MAY NOT WORK. FEEL FREE TO MODIFY
    def joinon(self, l_table, r_table, left_join=False):
        for leftrec in l_table:
            for rightrec in r_table:
                if self._on(leftrec,rightrec):
                    yielded = True
                    yield self.joinedfields(leftrec + tuple(rightrec))
                if left_join and not yielded:
                    yield self.joinedfields(leftrec + self._r_fields0)
                    
    def joinusing(self, l_table, r_table, left_join=False):
        right_grouper = groupby(sorted(r_table,
                                       key=self._r_using_key),
                                key=self._r_using_key)
        grouped_r_table = dict([(k,list(g)) for k,g in right_grouper])
        for leftrec in l_table:
            # get list of corresponding records from r_table
            rightrecs = grouped_r_table.get(self._l_using_getter(leftrec),
                                            [])
            if rightrecs:
                for rightrec in rightrecs:
                    yield self.joinedfields(leftrec + tuple(rightrec))
            else:
                if left_join:
                    yield self.joinedfields(leftrec + self._r_fields0)
  
    @staticmethod
    def concat(leftmodel, rightmodel):
        concat_tuples = []
        [concat_tuples.append(('left.'+t[0],t[1])) 
         for t in leftmodel.model_tuples]
        [concat_tuples.append(('right.'+t[0],t[1])) 
         for t in rightmodel.model_tuples]
        return TableModel(concat_tuples)
            
    @staticmethod 
    def result_model_tuples(lmodel, rmodel, 
                            selected_flds, 
                            qualified, unqualified):
        ''' 
        Go over the selected fields and ensure that qualified column names
            exist in respective model 
        raise if columns are not found or duplicate names encountered.
        return validated list of columns as well as unqualified list
        '''
        result_model_tuples = []
        for field in selected_flds:
            found = False
            try: 
                qual, unqual = lmodel.matchfield(field, 'left.')
                result_model_tuples.append( (unqual,
                                             lmodel.fieldformatter(unqual)) )
                found = True
            except:
                qual,unqual = rmodel.matchfield(field, 'right.')
                result_model_tuples.append( (unqual, 
                                             rmodel.fieldformatter(unqual)) )
                found = True
            if found:
                if unqual in unqualified:
                    # must not have duplicate columns in the joined table
                    raise JoinModelErr(fields=unqual,
                                       model=','.join(selected_flds))
                else:
                    unqualified.append(unqual)
                qualified.append(qual)
        return result_model_tuples
    
    def __repr__(self):
        qual_selects_str = ', '.join(self.__repr['qual_selects'])
        self.__repr['select'] = 'select {0}'.format(qual_selects_str)
        if self.isJoinOn():
            self.__repr['using_on'] = \
                'ON ({0})'.format(', '.join(self.__repr['ons']))
        else:
            self.__repr['using_on'] = \
                'USING ({0})'.format(', '.join(self.__repr['usings'])) 
        
        ret =   """
                {select}
                FROM left 
                JOIN/LEFT JOIN right {using_on}
                """.format(**self.__repr)
        return super(JoinModel, self).__repr__() + dedent(ret)

    def __str__(self):
        return self.__repr__()

def join(lefttable, righttable, joinmodel):
    '''
    Usage:
    left_right_model=TableModel([('oprid', gi), ('catnm', gs), ('grss',gd2)])
    join_model=JoinModel(leftmodel=left_right_model,
                         rightmodel = left_right_model, 
                         select=['left.grss',
                                 'right.oprid',
                                 'right.grss',
                                 'catnm'],
                         using=['oprid','catnm'])
    join(lefttable=[(0,'a',0.0), (1,'a',1.1), 
                     (0,'b',2.2), (1,'b',3.3), 
                     (1,'c',4.4)],
         righttable=[(0,'a',0.1), (1,'a',1.2), 
                      (0,'b',2.3), (1,'b',3.4)], 
         joinmodel = join_model
    --> result: 
    [(Decimal('0.00'), 0, Decimal('0.10'), 'a')
     (Decimal('1.10'), 1, Decimal('1.20'), 'a')
     (Decimal('2.20'), 0, Decimal('2.30'), 'b')
     (Decimal('3.30'), 1, Decimal('3.40'), 'b')]
    '''
    return list(joinmodel.join(lefttable, righttable))

def leftjoin(lefttable, righttable, joinmodel):
    '''
    left_right_model=TableModel([('oprid', gi), 
                                 ('catnm', gs), 
                                 ('grss',gd2)])
    join_model=JoinModel(leftmodel=left_right_model,
                         rightmodel = left_right_model, 
                         select=['left.grss',
                                 'right.oprid',
                                 'right.grss',
                                 'catnm'],
                         using=['oprid','catnm'])
    leftjoin(lefttable=[(0,'a',0.0), (1,'a',1.1), 
                         (0,'b',2.2), (1,'b',3.3), 
                         (1,'c',4.4)],
             righttable=[(0,'a',0.1), (1,'a',1.2), 
                          (0,'b',2.3), (1,'b',3.4)], 
             joinmodel = join_model
    --> result: 
    [(Decimal('0.00'), 0, Decimal('0.10'), 'a')
     (Decimal('1.10'), 1, Decimal('1.20'), 'a')
     (Decimal('2.20'), 0, Decimal('2.30'), 'b')
     (Decimal('3.30'), 1, Decimal('3.40'), 'b')
     (Decimal('4.40'), 0, Decimal('0.00'), '')]
    '''
    return list(joinmodel.leftjoin(lefttable, righttable))

class TabulateModel(AggModel):
    '''
    Tabulation Model
    Encapsulates data and operations needed to access and tabulate a given
        table-set, whose model is from_model.
    Extends AggModel since a group_by clause may be necessary if the table-set
        is not balanced and hasn't been aggregated
    '''
    # default parameters for tabulate
    table_writer = None
    tabulation_options = dict()
    TOTAL = '[TOTAL]'
    def __init__(self, 
                 from_model,
                 specs,
                 table_writer=None, **kwargs):
        self.table_writer = table_writer or TabulateModel.table_writer
        kwargs = dict(TabulateModel.tabulation_options, **kwargs)

        # field spec tuples may have 1 to 4 elements
        #    if 1 element, it will have an element or formula name only
        #    if 2 elements, it will have header, target and span = 1
        #    if 3 elements, it will have header, target and span
        #    if 4 elements, it will have header, target, span and footer
        # first selected field is treated as label (default footer 'TOTAL') 
        # second selected field onward are assumed to be summing fields unless 
        #     4th element (footer) is provided. They are flagged footer = sum
        # selected formula fields will be found in self._formulae
        # other selected fields are assumed to be data fields that may
        # be referenced in formulae but won't be totaled
        assert(specs and hasattr(specs, '__iter__') and specs[0])
        self.field_specs = dict()
        self.selected_refs = []
        self.selected_fields = []
        self.selected_sums = []
        self.selected_formulae = []
        self.selected_data = []
        self.selected_groupbys = []
        self.selected_callables = []
        for spec in specs:
            if not hasattr(spec, '__iter__'): spec = (spec,)
            if len(spec) == 1:
                ref, = spec
                header, footer, span = '','',0#will be hidden
            elif len(spec) == 2:
                header, ref = spec
                footer, span = '',1
            elif len(spec) == 3:
                header, ref, span = spec
                # selected_refs will be [] when label reference is processed
                footer = self.selected_refs and sum or TabulateModel.TOTAL
            elif len(spec) == 4:
                header, ref, span, footer = spec
            assert(isinstance(header, str) and
                   (isinstance(ref, str) or hasattr(ref,'__call__')) and
                   span == int(span) and#must be integer
                   (isinstance(footer, str) or footer == sum))
            # maintain a list of selected fields
            self.selected_refs.append(ref)
            if ref in from_model._formulae.keys():
                # spec references a formula field
                self.selected_formulae.append(ref)

            self.field_specs[ref] = dict(R=ref,H=header,S=span,F=footer)
            # resolve all fields referenced directly or indirectly in the specs
            # we need this step since references to formulae may require 
            # fields that are not explicitely requested in the tabulation
            # but, nevertheless, need to be made available for the formulae
            # to be applied successfully
            _ref_call, _ref_fields = from_model.resolvereference(ref)
            if _ref_fields:
                #spec references one or more model fields
                for f in _ref_fields:
                    if f not in self.field_specs:
                        self.field_specs[f] = dict(R=f,H='',S=0,F=footer)
                    self.selected_fields.append(f)
                if len(self.selected_refs) == 1:
                    # first reference is assumed to be the label
                    # label field values must be unique: must group by them
                    self.selected_groupbys.append(f)
                    
                if footer == sum:# the field will be subject to summing
                    self.selected_sums.append(f)
                else:# field is required but won't be summed 
                    self.selected_data.append(f)
            elif _ref_call: #must be a no-arg callable -> can't group by
                self.selected_callables.append(_ref_call)
            else:#should never be here
                assert(False)
        # must be a non-empty list of field names
        assert(self.selected_refs and self.selected_fields)
        if self.selected_groupbys:# we have something to groupby-aggregate on:
            _order_by = '{{{0}}}'.format(','.join(self.selected_groupbys))
            _group_by = self.selected_groupbys
            # aggregation params have been resolved, initialize agg model
            super(TabulateModel, self).__init__(from_model = from_model,
                                                select=self.selected_fields,
                                                sums=self.selected_sums,
                                                order_by=_order_by,
                                                group_by=_group_by)
        else:
            # no field to group-by on, initialize AggModel without aggregation 
            super(TabulateModel, self).__init__(from_model = from_model,
                                                select=self.selected_fields)
        # optional/keyword args:
        self.expand = not kwargs.get('collapse', False)
        self.entity_headers = kwargs.get('entity_headers', False)
        self.field_headers = kwargs.get('field_headers', False)
        self.summary_column = kwargs.get('summary_column', False)
        self.summary_row = kwargs.get('summary_row', False)
        self.data_format = kwargs.get('data_format', self.table_writer.DATA)
        formulae_tuples = self.getfromformulae()
        if formulae_tuples:
            self.defineformulae(formulae_tuples)
        self.compiled_formulae, self.formulae_formatters = \
            self.compileformulae(self.selected_formulae, self)

    def getfromformulae(self):
        '''Returns selected formulae spec tuples from self.frommodel'''
        return [(f[0],f[1]['F'],f[1]['D'])
                for f in self.frommodel._formulae.items()
                if f[0] in self.selected_formulae]

        
    def totals(self, fields, table, from_model = True):
        '''
        Given a field name, finds the total of that field in all records 
            and returns it in field format.
        @param table: list of tuples representing query result for a single
            entity. table and self.frommodel must be compatible
        @param from_model: whether the table follows the from-model of 
            the original table set being tabulated or the tabulation model
        '''
        def caller(f, v): return f(v)#returns a call on f with v as arg
        model = from_model and self.frommodel or self
        if not all(map(model.fieldexists, fields)):
            raise TableModelErr(fields=fields, model=model)
        _totals = []
        for f in fields:#sum values of each column
            ccvvs = model.columngetter(f, table)
            #sum summing fields, use footer for data fields
            if f in self.selected_sums:
                try:
                    cttl = sum(ccvvs)
                except:
                    #if for some reason, sum fails we take the default value
                    cttl = self.fieldformatter(f)(None)
                    
            elif self.field_specs[f]['F']:#must be a data field with a footer
                try:#assume callable
                    cttl = self.field_specs[f]['F'](ccvvs)
                except:#treat as string
                    cttl = self.field_specs[f]['F']
            else:#default field value
                cttl = self.fieldformatter(f)(None)
            _totals.append(cttl)
        # format totals using respective field formatters
        # otherwise we just return a list of totals
        formatted_totals = map(caller,
                               model.fieldformatters(fields),
                               _totals)
        return formatted_totals

    def applyformula(self, compiled_formula, formatter, record):
        return formatter(eval(compiled_formula)) 
    
    def open_row(self, row_format=None):
        row_format = row_format or self.data_format
        self.table_writer.open_row(row_format)
    
    def close_row(self):
        self.table_writer.close_row()
        
    def open_data_row(self):
        self.open_row(self.data_format)
    
    def open_summary_row(self):
        self.table_writer.open_summary_row()
         
    def add_data_cells(self,record, show_label = False):
        for i,f in enumerate(self.selected_refs):
            if f in self.selected_formulae:
                _formatted_value = self.applyformula(self.compiled_formulae[f],
                                                     self.formulae_formatters[f],
                                                     record)
            elif f in self.selected_callables:
                _formatted_value = f(record)
            else:
                _formatted_value = self.formattedgetter(f, record)
            if self.field_specs[f]['S']:
                if (i == 0) and show_label:
                    #first reference is assumed to be the label
                    self.table_writer.add_cell(_formatted_value,
                                               self.field_specs[f]['S'])
                elif i > 0:
                    self.table_writer.add_cell(_formatted_value,
                                               self.field_specs[f]['S'])
    def add_entity_header_row(self, entities): 
        self.table_writer.open_entity_header_row()
        label_specs = self.field_specs[self.selected_refs[0]]
        # empty cell
        self.table_writer.add_cell('',label_specs['S'])
        margin = len(self.selected_refs[1:]) * self.table_writer.FSS - self.table_writer.TSS
        _entity_header_span = sum([self.field_specs[f]['S'] 
                                   for f in self.selected_refs[1:]]) + margin

        # repeat as many entities as we have + summary column
        for entity in entities:
            self.table_writer.add_cell(entity, _entity_header_span)
        if self.summary_column: 
            self.table_writer.add_cell(label_specs['F'],
                                       _entity_header_span)
        self.table_writer.close_row()
          
    def add_field_header_row(self, entities):
        self.table_writer.open_field_header_row()
        label_specs = self.field_specs[self.selected_refs[0]] 
        self.table_writer.add_cell(label_specs['H'],
                                   label_specs['S'])
        for _ in entities:
            for f in self.selected_refs[1:]:
                if self.field_specs[f]['S']:
                    self.table_writer.add_cell(self.field_specs[f]['H'], 
                                               self.field_specs[f]['S']) 
        if self.summary_column:
            # and one last time if we're showing the totals' column
            for f in self.selected_refs[1:]:
                if self.field_specs[f]['S']:
                    self.table_writer.add_cell(self.field_specs[f]['H'],
                                               self.field_specs[f]['S']) 
        self.table_writer.close_row()
        
class TableSetErr(ValueError):
    INVALID_ARG0 = 'Must have model, model tuples or ValueSet in arg0'
    def __init__(self, err=INVALID_ARG0):
        super(TableSetErr, self).__init__(err)
class TableSet(object):
    '''
    Encapsulates one or more "tables" as well as common query-like 
        operations and printing on these tables as though on a single table
    Each table is a list of tuples returned by pgqueryobject.getresult().
        and is associated with an entity (ex: outlet) targetted by the query  
    pgqueryobjects result must share a model similar to TableSet.model
    Usage: a report that runs a given query once for each outlet 
        and stores individual pgqueryobject.getresult().
        post-query processing can then be done on all query results 
        as though on a single table. 
    '''
    def __init__(self, *args,**kwargs):#tables=[],table0=[]
        '''
        @param args: can be one of the following:
            - a single TableModel, 
            - a list of field specs to create a TableModel
            - a TableSet from which to shallow-copied _tables and entities
            - a list of ValueSets to create a TableModel and list of entities
        @keyword table0: default table (list of tuples) if @tables not provided
        @keyword tables: list of tuples: [(entityi,[(recordi,),]),]
            if @tables provided @table0 is ignored
        @keyword WHERE: a string to compile into a callable for filter()
        '''
        tables = None
        arg0 = args and args[0] or None
        if isinstance(arg0,TableModel):
            self.model = arg0
        elif isinstance(arg0, list): # assume list of model tuples
            self.model = TableModel(arg0)
        elif isinstance(arg0, TableSet):# TableSet
            self.model = arg0.model
            kwargs['tables'] = [(ent,arg0.gettable(ent)) 
                                for ent in arg0.entities]
#             kwargs['formulae'] = arg0._formulae.items()
        elif isinstance(arg0, ValueSet):# ValueSet
            # from value sets to list of values
            _vvss = ValueSet.callonsets('get', *args)
#             print '_vvss: ',_vvss#Nabil
            kwargs['tables'] = zip(*_vvss) #zip entities individually
            kwargs['tables'] = [(ent, [kwargs['tables'][i]])
                                for i, ent in enumerate(arg0.entities)
                                if ent]# filter out the None/total entity
            # combine their model tuples into a new model
            stt = [vs.model.model_tuples[0] for vs in args]
            self.model = TableModel(stt)
        elif isinstance(arg0, str):# comma-separated fields
            fields = [f.strip() for f in arg0.split(',')]
            # must provide one formatter per field or no formatters at all
            formatters = args[1:] or []
            assert(not formatters or len(formatters) == len(fields))
            # len(formatters)<=len(fields) always true
            flds_fmts = [t for t in izip_longest(fields,
                                                 formatters,
                                                 fillvalue=identity)]
            self.model= TableModel(flds_fmts)
        else:# not supported
            raise TableSetErr()
        
        # find param tables in kwargs
        # ignore @table0 if @tables is provided
        tables = tables or kwargs.get('tables',[])
        table0 = kwargs.get('table0')
        tables = tables or (table0 and [(None,table0)]) or []
        entities = [ent for ent, _ in tables]
#         # map formulae names and definitions
#         self._formulae = dict()
#         self.defineformulae(kwargs.get('formulae',[]))
        tables = dict(tables)
        # if we have a where clause, we apply it via AggModel
        if kwargs.get('WHERE'):
            where_cl_model = AggModel(from_model=self.model,
                                      select=self.model.listfields(),
                                      where=kwargs['WHERE'])
            for e,t in tables.items():
                tables[e] = where_cl_model.where(t)
            
        self._hasdata = False
        self._balanced = True
        self._tables = dict()
        self._entities = list()
        for entity in entities: self.addtable(entity, tables[entity])
        self._sub_model = None
   
    def __add__(self, table_set):
        '''
        Append entity tables from an existing TableSet
            to corresponding entity tables in this TableSet 
            and return resulting tableset
            The TableSets must have matching models
            entities that do not exist in this TableSet will be ignored
        '''
        assert(self.model == table_set.model)# must have equivalent models
        new_table_set = TableSet(self.model)
        for ent in TableSet.commonentities(self, table_set):
            new_table_set.addtable(ent, 
                                   self._tables[ent] + table_set._tables[ent])
        if self.balanced and table_set.balanced: 
            new_table_set._balanced=True
        return new_table_set
         
    @property
    def table0(self):
        return self.gettable(None)
    @property
    def entities(self):
        return self._entities
    @property
    def hasdata(self):
        return self._hasdata
    @property
    def balanced(self):
        '''
        Returns whether a table set is made up of balanced entity tables
            A table set is considered balanced if it:
            - has a uniform record count of 1
            - has only one entity table, 
            - is an aggregated table set
            - is made up of ValueSets,
            - is a concatenation of balanced TableSets 
        '''
        return self._balanced
            
    def addtable(self, entity, table):
        ''' add entity and map table (list of tuples) to it'''
        if entity == None or self.table0:
            # self.table0 may not co-exist with any other tables in a given set  
            self._tables = dict()
            self._entities = list()
        self._tables[entity]= table
        self._entities.append(entity)
        _table_has_data = table and True or False # does it have data?
        # update has-data and balanced properties
        self._hasdata = self._hasdata or _table_has_data
        self._balanced = len(self.entities) < 2 or self.recordcount() == 1
        
    def gettable(self, entity):
        ''' return list of tuples mapped to entity'''
        if self._tables.has_key(entity):
            if isinstance(self._tables[entity], list):
                return self._tables[entity]
            else:# assume pgqueryobject
                return self._tables[entity].getresult()
        else:
            return self._tables.get(None)
    
    def recordcount(self):
        '''
        Returns uniform record count. 
        If tables contain different record counts return 0
        '''
        # if we have no records, count is 0
        if not self._hasdata: return 0
        _recordcount = [len(self.gettable(entity)) for entity in self.entities]
        if min(_recordcount) == max(_recordcount):
            # equal record counts accross tables
            return _recordcount[0]
        else: 
            return 0
    def cleanformulae(self): 
        self.model.cleanformulae()
    def defineformulae(self, formula_tuples): 
        self.model.defineformulae(formula_tuples)
            
    def aggregate(self, **kwargs):
        if 'agg_model' in kwargs:
            agg_model = kwargs['agg_model']
        else:
            agg_model = AggModel(self.model,**kwargs)
        agg_set = TableSet(agg_model)
        agg_set._balanced = True
        agg_dicts = dict.fromkeys(self.entities)
        for entity in self.entities:
            agg_dict = agg_model.aggregate(self.gettable(entity))
            agg_dicts[entity] = agg_dict
        for entity in self.entities:
            #add padding to ensure entity tables are in sync
            agg_table = agg_model.pad(agg_dicts[entity]).values()
            agg_set.addtable(entity,agg_model.order(agg_table))
        return agg_set
    
    def record_vectors(self, fields=[]): 
        '''
        Yields a list of records each of which is an ordered concatenation 
            of corresponding records from all tables.
        The yielded record vector will only have the selected fields
        If no fields are specified, corresponding records are appended as-is
            into the yielded record vector.
        Usage: 
            tableset.record_vectors(_fields)
        '''
        if fields:
            # create a single getter for all fields
            _fields_getter = self.model.fieldsgetter(fields, 
                                                     tuplize=True)
        else:
            # if no fields specified, we return entire records
            # in the table sets model
            _fields_getter = identity
        _recordcount = self.recordcount()
        rrv = [] # concatenation of ith record from each entity 
        for i in range(_recordcount):
            # put corresponding entity records into a single list of tuples
            rrv = [_fields_getter(self.gettable(entity)[i]) 
                   for entity in self.entities]
            yield rrv
            
    def total_sets(self, fields, entity=None):
        '''
        Returns value sets for all fields
            if fields is a Aggregation/Tabulation model, 
            returns value sets of all fields in that model
        NOTE: AggModel.from_model must be compatible with the this table set 
        '''
        entities = [entity] if entity else self.entities
        if isinstance(fields, AggModel):
            model = fields
            fields = model.listfields()
        else:# assume table set's model
            model = self.model
            if not fields:
                fields = model.listfields()
        if not hasattr(fields, '__iter__'): fields = [fields]
        #Map field name to its ValueSet
        fftt = []# list of field totals (list(ValueSet))
        #initialize all field sets to their properly typed/formatted value 
        for f in fields:
            _formatter = model.fieldformatter(f)
            fftt.append(ValueSet(f,_formatter))
        for ent in entities:
            ett = model.totals(fields, self.gettable(ent))
            for i, _ in enumerate(fields):
                fftt[i].add((ent, ett[i]))
        return len(fftt) == 1 and fftt[0] or fftt #

    def totals(self, fields, entity=None):
        entities = [entity] if entity else self.entities
        if hasattr(fields, '__iter__'):
            # multiple fields: each entity will have a dictionary 
            #   of fields and their totals initialized to None
            mkdefaultdict = partial(dict.fromkeys,fields,None)
            entity_total_dict = defaultdict(mkdefaultdict)
            #initialize all totals to their properly typed/formatted zero value  
            for f in fields:
                for ent in entities + [None]:
                    entity_total_dict[ent][f] = \
                        self.model.fieldformatter(f)(None) 
        else:
            # one field only: each entity will have a single total
            # also combine totals under key None
            mkdefaultdict = partial(self.model.fieldformatter(fields), None)
            # dict of entities and their totals
            entity_total_dict = defaultdict(mkdefaultdict)
            fields = [fields] # we accept single field name but use as list
        for t in entities:
            entity_totals = self.model.totals(fields, self.gettable(t))
            if len(fields) > 1:
                # if more than one field, each entity will have a dictionary 
                # of field-totals
                for i, field in enumerate(fields):
                    entity_total_dict[t][field] = entity_totals[i]
                    entity_total_dict[None][field] += entity_totals[i]
            else:# only one field: each entity gets a single field total
                entity_total_dict[t] = entity_totals[0]
                entity_total_dict[None] += entity_totals[0]
        return entity_total_dict

    
    def join(self, rightset, leftjoin=False,**kwargs):
        '''
        Reminiscent of a basic SELECT-JOIN-USING and SELECT-JOIN-ON 
            psql query but operates on two table-sets instead 
            of actual psql tables.
        It supports basic ON clause. If no ON clause is provided, 
            USING kw must be provided
        '''
        join_model = JoinModel(self.model, rightset.model,**kwargs)
        joined_set = TableSet(join_model)
        # iterate over common entities
        entities = TableSet.commonentities(self, rightset)
        for en in entities:
            # join table pairs for each entity. 
            # if, for a given entity either tables is missing,
            #     we look for a common table under None key. 
            # Otherwise, we pass an empty table
            if leftjoin: 
                joined_table = list(join_model.leftjoin(self.gettable(en), 
                                                        rightset.gettable(en)))
            else:
                joined_table = list(join_model.join(self.gettable(en),
                                                    rightset.gettable(en)))
            joined_set.addtable(en,joined_table)
        return joined_set

    def leftjoin(self, right_set, **kwargs):
        return self.join(right_set, leftjoin=True, **kwargs)
   
#     def applyformula(self, compiled_formula, formatter, record):
#         return formatter(eval(compiled_formula)) 
    
    def tabulate(self,field_specs,table_writer=None,**kwargs):
        '''
        @param field_specs: List of tuples specifying field header, 
            name and span
        Usage: 
        grp_ttls.tabulate(field_specs= [('[VXL_GROUPING]','catgrpnm',1)
                                        ('[QUANTITY]','qty',2),
                                        ('[AMOUNT]','grss',1)], 
                          table_writer = output, **options)
        @param table_writer: An instance of TableWriter
        Optional args:
        @param collapse = skip writing actional data rows
        @param entity_headers = write entity headers row
        @param field_headers = write field headers row
        @param summary_column = calculate and write the summary 
            of each record vector in the last column 
        @param summary_row = calculate and write the summary 
            of each entity in the last row
            when used with summary_column also writes summary 
            of all entities and keys in the bottom right corner. 
        ==>
          ===>>>|        |  StoreName1    |  StoreName2    |      Total     |
          ===>>>|GROUPING| QUANTITY|AMOUNT| QUANTITY|AMOUNT| QUANTITY|AMOUNT|
                =============================================================
          ===>>>|Group1  |        X| XX.XX|        X| XX.XX|        X| XX.XX
          ===>>>...
          ===>>>|Groupn  |        X| XX.XX|        X| XX.XX|        X| XX.XX
                =============================================================
          ===>>>|Total   |        X| XX.XX|        X| XX.XX|        X| XX.XX
        '''
        # create a new tabulation/aggregation 'submodel' 
        #     for fields selected for tabulation
        _tabul_model = TabulateModel(self.model,
                                     field_specs,
                                     table_writer,**kwargs)
        if _tabul_model.entity_headers: 
            _tabul_model.add_entity_header_row(self.entities)
        if _tabul_model.field_headers: 
            _tabul_model.add_field_header_row(self.entities)
        if _tabul_model.expand:
            # ensure that entity tables are balanced/padded
            if self.balanced:
                _balanced = self
                record_vectors = _balanced.record_vectors(_tabul_model.listfields())
            else:# entity tables need to be made balanced/padded
                # do label specs reference a field that we can group on?
                if _tabul_model.selected_groupbys:
                    # label fields have been resolved: we can group by them
                    _balanced = self.aggregate(agg_model=_tabul_model)
                    # we get the vector generator for all fields in tabulation
                    record_vectors = _balanced.record_vectors()
                else:
                    # nothing to aggregate on, 
                    #     tabulate without balancing at caller's risk 
                    record_vectors = self.record_vectors(_tabul_model.listfields())
                    
            for _rrv in record_vectors:
                _tabul_model.open_data_row()
                for i, record in enumerate(_rrv):
                    # show label for first record in each record vector
                    show_label= (i == 0) and True or False
                    _tabul_model.add_data_cells(record,show_label=show_label)
                # now for the totals column for this row
                if _tabul_model.summary_column:
                    _record_vectors_total = \
                        _tabul_model.totals(_tabul_model.listfields(),
                                            _rrv,
                                            from_model = False)
                    _tabul_model.add_data_cells(_record_vectors_total)
                _tabul_model.close_row()
            
        # now for the totals' row
        if _tabul_model.summary_row:
            _tabul_model.open_summary_row()
            _total_sets = self.total_sets(_tabul_model)
            if not hasattr(_total_sets, '__iter__'): 
                _total_sets = [_total_sets]
            if _total_sets:
                # get all values in each value set in form of tuples
                _vvtt=ValueSet.callonsets('get', *_total_sets)
                _rrvv = zip(*_vvtt) #zip entities individually
            for i, _rrv in enumerate(_rrvv):
                show_label= (i == 0) and True or False
                _tabul_model.add_data_cells(_rrv,show_label=show_label)
            if _tabul_model.summary_column:
                # calculate combined total in each set under key None
                ValueSet.callonsets(('add', None), *_total_sets)
                # make tuple out of combined totals
                _vvtt = ValueSet.callonsets(('get', None), *_total_sets)
                _tabul_model.add_data_cells(_vvtt)
            _tabul_model.close_row()
            _total_sets = [s for s in _total_sets
                           if s.name in _tabul_model.selected_refs and
                                s.name in _tabul_model.selected_sums]
            return _total_sets
    
    def summarize(self,field_specs,table_writer=None,**kwargs):
        '''
        @param field_specs: List of tuples specifying field header, name 
            and span
        Usage: 
        grp_ttls.summarize(field_specs= [('[VXL_GROUPING]', 'catgrpnm', 1)
                                         ('[QUANTITY]','qty', 2),
                                         ('[AMOUNT]','grss', 1)],
                           table_writer = output)
        ==>
                |        |  StoreName1    |  StoreName2    |      Total     |
          ===>>>|GROUPING| QUANTITY|AMOUNT| QUANTITY|AMOUNT| QUANTITY|AMOUNT|
                =============================================================
                                            HIDDEN
                =============================================================
          ===>>>|Total   |        X| XX.XX|        X| XX.XX|        X| XX.XX
        '''
        kwargs.update(collapse=True)
        return self.tabulate(field_specs,
                             table_writer or TabulateModel.table_writer,
                             **kwargs)
    
    def condense(self,field_specs,table_writer=None,**kwargs):
        '''
        @param field_specs: List of tuples specifying 
            field header, name and span
        Usage: 
        grp_ttls.condense(field_specs= [ ('[VXL_GROUPING]', 'catgrpnm', 1)
                                         ('[QUANTITY]','qty', 2),
                                         ('[AMOUNT]','grss', 1)], 
                          table_writer = output)
        ==>
                |                        HIDDEN HEADER                   |
                ==========================================================
          ===>>>|Group   |       X| XX.XX|       X| XX.XX|       X| XX.XX
                ==========================================================
                |                        HIDDEN SUMMARY                  |
        '''
        table_writer = table_writer or TabulateModel.table_writer
        kwargs.update(field_headers = False, 
                      summary_row = False, 
                      data_format = table_writer.CONDENSE)
        return self.tabulate(field_specs, table_writer, **kwargs)
    
    def __repr__(self):
        getrec = lambda r: '\t'.join(map(repr,r))
        _repr = self.model.__repr__() + '\n'
        for entity, table in self._tables.items():
            _repr += 'table {0}:\n'.format(entity)
            _repr += '\n'.join(map(getrec,table)) + '\n' 
        return _repr
    
    def __str__(self):
        getrec = lambda r: '\t'.join(map(str,r))
        _str = self.model.__str__() + '\n'
        for entity, table in self._tables.items():
            _str += 'table {0}:\n'.format(entity)
            _str += '\n'.join(map(getrec,table)) + '\n' 
        return _str

    @staticmethod
    def commonentities(*tablesets):
        ''' return entities common among all given TableSets '''
        if len(tablesets) > 2:
            reduce(TableSet.entities, tablesets)
        elif len(tablesets) == 2:
            table1, table2 = tablesets
            if table1.table0 and table2.table0:
                return [None]
            elif table1.table0 and table2.entities:
                return table2.entities
            elif table1.entities and table2.table0:
                return table1.entities
            elif table1.entities and table2.entities:
                return filter(lambda ent: ent in table2.entities,
                              table1.entities) 
            else:
                return []
    
class ValueSetErr(ValueError):
    VALSET_MISMATCH = 'ValueSet operands must have same list of entities'
    def __init__(self, err=VALSET_MISMATCH): 
        super(ValueSetErr, self).__init__(err)

class ValueSet(object):
    def __init__(self,field,*args):
        assert(field)
        values = args
        if isinstance(field, tuple):#model tuple
            self.model = TableModel([field])
        elif isinstance(field, str):#field name
            #does args have a formatter?
            try:
                if hasattr(args[0],'__call__'):
                    formatter = args[0]
                    values = args[1:]#values are after formatter
                else:
                    formatter = identity
            except:
                formatter = identity
            self.model = TableModel([(field,formatter)])
        self.name = self.model.fieldname(0)
        self.formatter = self.model.fieldformatter(self.name)
        self._entities = []
        self._values = {}
        if values:
            values = values[0]
        else:
            return# no values to add yet
        if isinstance(values, list):
            self._values = dict([(ent, self.formatter(val)) 
                                 for ent, val in values])
            self._entities = [val[0] for val in values]
        elif isinstance(values, ValueSet):
            if self.formatter <> values.formatter:
                _values = [(ent, self.formatter(val)) 
                           for ent, val in values.getnext()]
            else:
                _values = [(ent, val) for ent, val in values.getnext()]
            self._values = dict(_values)
            self._entities = [ent for ent, val in _values]
        else:
            self._v0 = self.formatter(values)
    
    @property
    def entities(self):
        return self._entities

    def add(self, ent_val):
        if ent_val:
            self._entities.append(ent_val[0]) 
            self._values[ent_val[0]] = self.formatter(ent_val[1])
        else:
            self._entities.append(None)
            vv = self._values.values()
            try:# try summing. If unsuccessful, return first none-null value
                self._values[None] = self.formatter(sum(vv))
            except:
                self._values[None] = self.formatter(next(first 
                                                         for first in vv 
                                                         if first is not None))

    def get(self, *ents):
        _ents = ents or self.entities or []
        if len(_ents) == 1:
            return self._values[_ents[0]]
        else:
            return tuple(self._values[ent] for ent in _ents)
    
    def asdict(self):
        return self._values
    
    def getnext(self):
        for ent in self.entities:
            yield (ent, self.get(ent))
    def __abs__(self): 
        return ValueSet.operation(operator.__abs__, self) 
    def __add__(self, value_set): 
        return ValueSet.operation(operator.__add__, self, value_set) 
    def __and__(self, value_set): 
        return ValueSet.operation(operator.__and__, self, value_set)
    def __div__(self, value_set): 
        return ValueSet.operation(operator.__div__, self, value_set) 
    def __floordiv__(self, value_set): 
        return ValueSet.operation(operator.__floordiv__, self, value_set)
    def __mod__(self, value_set): 
        return ValueSet.operation(operator.__mod__, self, value_set)
    def __mul__(self, value_set): 
        return ValueSet.operation(operator.__mul__, self, value_set)
    def __neg__(self): 
        return ValueSet.operation(operator.__neg__, self)
    def __or__(self, value_set): 
        return ValueSet.operation(operator.__or__, self, value_set)
    def __pos__(self): 
        return ValueSet.operation(operator.__pos__, self)
    def __pow__(self, value_set): 
        return ValueSet.operation(operator.__pow__, self, value_set)
    def __sub__(self, value_set): 
        return ValueSet.operation(operator.__sub__, self, value_set)
    def __truediv__(self, value_set): 
        return ValueSet.operation(operator.__truediv__, self, value_set)

    @staticmethod
    def operation(*args, **kwargs):
        _operator = args[0]
        _operand1 = args[1]
        _operand1.add(None) #ensure operand has a None/total entity
        
        if len(args) == 3: # binary operation
            _operand2 = args[2]
            _operand2.add(None) #ensure operand has a None/total entity
            formatter = bestformat(_operand1.formatter,_operand2.formatter)
            name = _operand1.name + _operator.__name__ + _operand2.name
            if hasattr(_operand1, '_v0') and hasattr(_operand2, '_v0'): 
                return ValueSet(name,formatter, 
                                _operator(_operand1._v0, _operand2._v0))
            elif hasattr(_operand1, '_v0'): 
                return ValueSet(name, formatter,
                                [(ent, _operator(_operand1._v0,
                                                 _operand2.get(ent))) 
                                 for ent in _operand2.entities])
            elif hasattr(_operand2, '_v0'):
                return ValueSet(name, formatter,
                                [(ent, _operator(_operand1.get(ent),
                                                 _operand2._v0)) 
                                 for ent in _operand1.entities])
            else:
                #first raise if operands are not compatible:
                if _operand1._values.keys() != _operand2._values.keys():
                    raise ValueSetErr()
                else:
                    return ValueSet(name, formatter, 
                                    [(ent, _operator(_operand1.get(ent),
                                                     _operand2.get(ent)))
                                     for ent in _operand1.entities])
        elif len(args) == 2: #unary operation
            if hasattr(_operand1._values,'_v0'):
                return ValueSet(_operand1.name, 
                                _operand1.formatter, 
                                _operator(_operand1._v0))
            else: #_operand1 hasattr values
                return ValueSet(_operand1.name, 
                                _operand1.formatter,
                                [(ent, _operator(_operand1.get(ent))) 
                                 for ent in _operand1.entities])
        else:
            raise ValueSetErr('expecting 2 or 3 arguments!')
        
    @staticmethod
    def callonsets(*args):
        _caller_params = args[0]
        if not hasattr(_caller_params, '__iter__'):
            _caller_params = [_caller_params]
        _sets = args[1:]
        _caller = methodcaller(*_caller_params)
        return map(_caller, _sets)

    def __repr__(self):
        _repr = 'ValueSet %s %s:\n' %(self.name, self.formatter)
        _repr += '\t'.join(str(ent or 'Total') for ent in self.entities) + '\n'
        _repr += '\t'.join([str(self._values[ent]) for ent in self.entities])
        return _repr
    
    def __str__(self):
        _str = self._values.__str__() + '\n'
        return _str

def applicable(*args):
    '''
    if args are str, concatenates none-empty args separated by 'and'
    if args are tuples, appends none-empty tuples to the returned list
    else, appends none
    '''
    if isinstance(args[0], str):
        return " and ".join(filter(partial(bool),args))
    elif isinstance(args[0], tuple):
        return filter(partial(bool),args)
    else:
        raise ValueError("Expected sequence of str or tuple, got {0}".\
                         format(args[0].__class__.__name__))

def post_sql(**kwargs):
    '''
    Maps generic sql-like calls to the somewhat less intuitive 
        syntax in corresponding calls in TableModel, TableSet, etc...
    '''
            
    # validate parameters and convert from string to list if need be
    # also adjust parameter names to in-module nomenclature
#     print id(post_sql._params['FROM'])
#     print id(post_sql.alias)
    validated= dict()
    try:
        _from_set = kwargs['FROM']
        kwargs['SELECT']
    except:
        raise ValueError('post_sql requires SELECT and FROM keyword arguments')
    for p in kwargs:
        try:
            expected = post_sql.expected(p)
            alias= post_sql.alias(p)
            got=kwargs[p].__class__
            value=kwargs[p]
        except: 
            msg='post_sql():Invalid keyword argument {0}'.format(p)
            raise SyntaxError(msg)
        else:
            if p == 'FROM' and expected == got:
                continue#already in _from_set
            elif expected == got: 
                validated[alias]=value
            elif expected == list and got == str:
                validated[alias]=list(f.strip() for f in value.split(','))
            else:
                msg='post_sql keyword {0}: Expected:{1}, got:{2}'.\
                      format(p, expected.__name__, got.__name__)
                raise TypeError(msg)
    else:#assume valid call, but join or aggregate?
        if 'join' in validated:
            _join_set = validated.pop('join')
            return _from_set.join(_join_set, **validated)
        elif 'leftjoin' in validated:
            _join_set = validated.pop('leftjoin')
            return _from_set.leftjoin(_join_set, **validated)
        else: # assume aggregation
            return _from_set.aggregate(**validated)
#maps post_sql kwargs to expected type and related post_query params
post_sql._params= dict(SELECT=(list,'select'),
                       SUM=(list,'sums'),
                       GROUPBY=(list,'group_by'),
                       USING=(list,'using'),
                       FROM=(TableSet,'fromset'),
                       JOIN=(TableSet,'join'),
                       LEFTJOIN=(TableSet,'leftjoin'),
                       ON=(str,'on'),
                       ORDERBY=(str,'order_by'),
                       WHERE=(str,'where'))
post_sql.expected=lambda p: post_sql._params[p][0]
post_sql.alias=lambda p: post_sql._params[p][1]

class Launcher(object):
    from sys import modules
    def __init__(self):
        self.module = Launcher.modules[__name__]
        self.optionparser = OptionParser()
        _, args = self.optionparser.parse_args()
        torun = self.select(*args)
        self.launch(*torun)
    
    def select(self, *args):
        # get all Unit test functions in current module
        test_funcs = filter(lambda fn: fn.startswith('test_'),
                            dir(self.module))
        torun = []
        if not args: args = ['all']
        for arg in args:
            if arg == 'all':# run all unit tests
                torun = test_funcs
                break
            elif arg in test_funcs:
                torun.append(arg)
            else:
                print """
                post_query Available Tests:\nall\n{0}""".format('\n'.join(test_funcs))
                return list()
        return torun
            
    def launch(self, *args):
        for arg in args:
            methodcaller(arg)(self.module)
            
        
if __name__ == '__main__':
    Launcher()
