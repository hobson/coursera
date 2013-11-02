from .util import listify, intify
from collections import Mapping
import json
import re

from django.conf import settings

# these are for the TestForm object in django-bootstrap 
from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.contrib import messages
from django.template.context import RequestContext
# from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
# from django.forms.formsets import formset_factory
from bootstrap_toolkit.widgets import BootstrapUneditableInput #, BootstrapDateInput, BootstrapTextInput, 
from web.forms import HomeForm

def home_form(request):
    messages.success(request, 'Successfully loaded a form?')
    layout = request.GET.get('layout')
    if not layout:
        layout = 'vertical'
    if request.method == 'POST':
        form = HomeForm(request.POST)
        form.is_valid()
    else:
        form = HomeForm()
    form.fields['title'].widget = BootstrapUneditableInput()
    return render_to_response('form.html', RequestContext(request, {
        'form': form,
        'layout': layout,
    }))


def is_valid_javascript_callback(javascript_function_name, regex=None, max_len=None):
    if max_len is None:
        max_len = 256
    if regex is None:
        regex = re.compile('^[a-zA-Z][\w.]{0,%s}*$' % (max_len - 1))
    return bool(regex.match(javascript_function_name))
# # a more permissive one that allows unicode and leading undercores and $ from http://stackoverflow.com/a/2008444/623735
# RE_IS_JS_NAME = re.compile(r'^[_$a-zA-Z\xA0-\uFFFF][_$a-zA-Z0-9\xA0-\uFFFF]*$')


class AJAXResponse(HttpResponse):

    def __init__(self, request, data):
        indent = 0
        mime_type = 'application/json'
        if settings.DEBUG:
            mime_type = 'text/javascript'
            indent = 2
        data_as_json = json.dumps(data, indent=indent)

        javascript_function_name = request.GET.get('callback')
        if javascript_function_name and is_valid_javascript_callback(javascript_function_name):
            data_as_json = '%s(%s);' % (javascript_function_name, data_as_json)

        # TODO: Find out if it's OK to return a value, or should I just instantiate the response
        return super(AJAXResponse, self)(mimetype=mime_type, content=data_as_json)

def parse_node_name(name, use_defaults=False):
    """
    >>> sorted(parse_node_name('Origin,2.7, 3 ')[1].items())
    [('charge', 2.7), ('group', 3), ('name', 'Origin')]
    >>> parse_node_name('Origin,2.7, 3 ')[0]
    'Origin'
    """
    # if the name is not a string, but a dict defining a node, then just set the defaults and return it
    if isinstance(name, Mapping):
        ans = dict(name)
        for j, field in enumerate(parse_node_name.schema):
            if field['key'] not in ans:
                ans[field['key']] = field['default']
        return ans
    seq = listify(name, delim=',')
    ans = {}
    for j, field in enumerate(parse_node_name.schema):
        if 'default' in field:
            try:
                ans[field['key']] = field['type'](seq[j])
            except:
                if use_defaults:
                    ans[field['key']] = field['default']
        else:
            try:
                ans[field['key']] = ans.get(field['key'], field['type'](seq[j]))
            except:
                pass
    return ans
parse_node_name.schema = (
                {'key': 'name', 'type': str},  # TODO: use the absence of a default value (rather than index > 0) to identify mandatory fields
                {'key': 'charge', 'type': float, 'default': 1},
                {'key': 'group', 'type': intify, 'default': 0},  # TODO: this should be a string like the names/indexes to nodes (groups are just hidden nodes)
              )


def node_name_dictionaries(edges):
    """
    Return 2 dictionaries that translate from the cleaned/striped name to fully qualified node names, and vice versa.
    """
    node_names_only = []
    for edge in edges:
        node_names_only += [parse_node_name(edge['source'])['name'], parse_node_name(edge['target'])['name']]
    node_names = list(set(node_names_only))
    new_nodes, old_nodes = {}, {}
    for new, old in enumerate(node_names):
        new_nodes[old] = new
        old_nodes[new] = old
    return old_nodes, new_nodes


def node_name_lists(edge_list):
    """
    Return 2 lists that retain the order of nodes mentioned in the edges list: a list of full names and a list of cleaned names.

    node_name_lists([{'source': 'Origin,2.7, 3 ', 'target': 'Destination,1,2', 'value': 9}, {'source': 'Origin,4', 'target': 'New', 'value': 1}])
    (['Origin,2.7, 3 ', 'Destination,1,2', 'New'], ['Origin', 'Destination', 'New'])
    """
    node_names_only, node_full_names = [], []
    for edge in edge_list:
        node_full_names += [edge['source'], edge['target']]
        node_names_only += [parse_node_name(node_full_names[-2])['name'], parse_node_name(node_full_names[-1])['name']]
    for_del = []
    for i, name in enumerate(node_names_only):
        if name in node_names_only[:i]:
            for_del += [i]
    for i in reversed(for_del):
        del(node_full_names[i])
        del(node_names_only[i])
    return node_full_names, node_names_only


def parse_node_names(edges):
    """
    >>> parse_node_names([{'source': 'Origin,2.7, 3 ', 'target': 'Destination,1,2', 'value': 9}, {'source': 'Origin,4', 'target': 'New', 'value': 1}])
    [{'charge': 4.0, 'group': 3, 'name': 'Origin'}, {'charge': 1.0, 'group': 2, 'name': 'Destination'}, {'name': 'New'}]
    """
    full_name, short_name = node_name_dictionaries(edges)
    redundant_nodes = []
    for edge in edges:
        redundant_nodes += [parse_node_name(edge['source'])] 
        redundant_nodes += [parse_node_name(edge['target'])]
    distinct_names, ans = [], []
    #print redundant_nodes
    for i, node in enumerate(redundant_nodes):
        if node['name'] not in distinct_names:
            distinct_names += [node['name']]
            ans += [node]
        else:
            # `name` and other mandatory fields have already been taken care of (and possibly nonmandatory fields), 
            # so skip mandatory fields (the ones without a default value) and update the nonmandatory ones if there's new data
            for j, field in enumerate(parse_node_name.schema):
                if 'default' in field and field['name'] in node:
                    ans[distinct_names.index(node['name'])][field['name']] = node.get[field['name']]
    for i, node in enumerate(ans):
        for j, field in enumerate(parse_node_name.schema):
            if 'default' in field and field['name'] not in ans[i]:
                ans[i][field['name']] = field['default']        
    return ans



def naively_parse_edge_string(edge_string):
    """
    >>> naively_parse_edge_string('Origin,3,9_I_5.6~Origin_II~I_III')
    #[{'source': 'Origin', 'target': 'I', 'value': 5.6}, {'source': 'Origin', 'target': 'II'}, {'source': 'I', 'target': 'III'}]
    [{'source': 'Origin,3,9', 'target': 'I', 'value': 5.6}, {'source': 'Origin', 'target': 'II', 'value': 4}, {'source': 'I', 'target': 'III', 'value': 4}]
    """
    triplets = listify(edge_string, delim=naively_parse_edge_string.delim)
    edge_list = []
    for i, row in enumerate(triplets):
        if not isinstance(row, basestring):
            continue
        rec = {}
        x = listify(row, delim=naively_parse_edge_string.node_pair_delim)
        for j, field in enumerate(naively_parse_edge_string.schema):
            try:
                rec[field['name']] = field['type'](x[j])
            except:
                rec[field['name']] = field['default']
        edge_list += [rec]
    return edge_list
naively_parse_edge_string.node_pair_delim = '_'
naively_parse_edge_string.delim = '~'
naively_parse_edge_string.schema = (
                        {'name': 'source', 'type': str, 'default': 0},
                        {'name': 'target', 'type': str, 'default': 0},
                        {'name': 'value', 'type': float, 'default': 4},
                     )


def parse_graph_definition(edge_string):
    """

    >>> parse_graph_definition('Origin,3,9_I_5.6~Origin_II~I_III')
    [{'source': 0, 'target': 1, 'value': 5.6}, {'source': 0, 'target': 2, 'value': 4}, {'source': 1, 'target': 3, 'value': 4}]
    """
    edge_definitions = listify(edge_string, delim=parse_graph_definition.delim)
    edge_list, node_list, node_names = [], [], []
    for i, edge_definition in enumerate(edge_definitions):
        edge = {}
        node_string_pair = listify(edge_definition, delim=parse_graph_definition.node_pair_delim)
        for j, field in enumerate(parse_graph_definition.schema):
            try:
                obj = field['type'](node_string_pair[j])
                # if the field holds a node (source or target) then its name must be retrieved
                if isinstance(obj, Mapping) and 'name' in obj:
                    # is this node name already in our list (and other node properties already defined)
                    if obj['name'] in node_names:
                        node_index = node_names.index(obj['name'])
                        # update the existing node with any new properties of this node
                        node_list[node_index].update(obj)  # obj[1] is a node dict
                        obj = node_index
                    else:
                        # reprocess/recast the object (which is now node dict) to add default values to the node
                        obj = field['type'](obj)
                        node_list += [obj]  # obj[1] is a node dict
                        node_names += [obj['name']]  # obj[1] is a node dict
                        # since this is a new node, its node index is one less than the len of the node list (or node name list)
                        obj = len(node_list) - 1
                edge[field['key']] = obj
            except:
                edge[field['key']] = field['default']
        edge_list += [edge]
    return edge_list, node_list
parse_graph_definition.delim = '~'
parse_graph_definition.node_pair_delim = '_'
parse_graph_definition.schema = (
                        {'key': 'source', 'type': parse_node_name, 'default': 0},
                        {'key': 'target', 'type': parse_node_name, 'default': 0},
                        {'key': 'value', 'type': float, 'default': 4},
                     )


def connections(request, edges):
    """
    Plot a force-directed graph based on the edges provided
    """
    edge_list, node_list = parse_graph_definition(edges)
    data = {'nodes': json.dumps(node_list), 'edges': json.dumps(edge_list)}
    return render_to_response('call_center/connections.html', data)
