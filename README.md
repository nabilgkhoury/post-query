INTRODUCTION:
post_query.py is a module providing psql SELECT-like behaviour on lists of tuples instead of table records. Use this module to facilitate re-organization, aggregation and tabulation of existing query results to achieve the desired presentation without having to introduce complex changes to the actual psql queries. This will keep the queries general enough to be reused in child reports leaving them room to re-organize and aggregate query results as per their own requirements

RECOMMENDED USAGE:
    1- Create a post_query.TableSet instance
    2- Call post_query.post_sql to carry out aggregation and/or joining
    3- Call TableSet.tabulate() or one of its variants to generate
        a tabular representation of the data.

DEMONSTRATION:
The provided code snippets, all functions prefixed with “test_”, test and demonstrate all major functionality in this module. They can be run at the console as follows:
$ python post_query.py [all | test_joins test_value ...]

You can also list available tests as follows:
 $ python post_query.py test