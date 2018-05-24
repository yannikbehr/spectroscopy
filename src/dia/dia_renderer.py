import re
import dia


class Klass:
    def __init__(self, name):
        self.name = name
        self.stereotype = None
        # use a list to preserve the order
        self.attributes = []
        # a list, as java/c++ support multiple methods with the same name
        self.operations = []
        self.comment = ""
        self.parents = []
        self.templates = []
        self.inheritance_type = ""

    def AddAttribute(self, name, type, visibility,
                     value, comment, class_scope):
        self.attributes.append((name, (type, visibility, value,
                                       comment, class_scope)))

    def AddOperation(self, name, type, visibility, params, inheritance_type,
                     comment, class_scope):
        self.operations.append((name, (type, visibility, params,
                                       inheritance_type, comment,
                                       class_scope)))

    def SetComment(self, s):
        self.comment = s

    def AddParrent(self, parrent):
        self.parents.append(parrent)

    def AddTemplate(self, template):
        self.templates.append(template)

    def SetInheritance_type(self, inheritance_type):
        self.inheritance_type = inheritance_type

    def SetStereotype(self, stereotype):
        self.stereotype = stereotype


class ObjRenderer:
    """
    Implements the Object Renderer Interface and transforms
    diagram into its internal representation
    """
    def __init__(self):
        self.klasses = {}
        self.arrows = []
        self.filename = ""

    def begin_render(self, data, filename):
        self.filename = filename
        # not only reset the filename but also the other state,
        # otherwise we would accumulate information through every export
        self.klasses = {}
        self.arrows = []
        for layer in data.layers:
            # for the moment ignore layer info. But we could use this
            # to spread accross different files
            for o in layer.objects:
                if o.type.name == "UML - Class":
                    # print o.properties["name"].value
                    k = Klass(o.properties["name"].value)
                    k.SetComment(o.properties["comment"].value)
                    k.SetStereotype(o.properties["stereotype"].value)
                    if o.properties["abstract"].value:
                        k.SetInheritance_type("abstract")
                    if o.properties["template"].value:
                        k.SetInheritance_type("template")
                    for op in o.properties["operations"].value:
                        # op : a tuple with fixed placing,
                        # see: objects/UML/umloperations.c:umloperation_props
                        # (name, type, comment, stereotype, visibility,
                        # inheritance_type, class_scope, params)
                        params = []
                        for par in op[8]:
                            # par : again fixed placement,
                            # see objects/UML/umlparameter.c:umlparameter_props
                            # (name, type, value, comment, kind)
                            params.append((par[0], par[1], par[2], par[3],
                                           par[4]))
                        k.AddOperation(op[0], op[1], op[4], params, op[5],
                                       op[2], op[7])
                        # print o.properties["attributes"].value
                    for attr in o.properties["attributes"].value:
                        # see objects/UML/umlattributes.c:umlattribute_props
                        # print "\t", attr[0], attr[1], attr[4]
                        # name, type, value, comment, visibility, abstract,
                        # class_scope
                        k.AddAttribute(attr[0], attr[1], attr[4], attr[2],
                                       attr[3], attr[6])
                    self.klasses[o.properties["name"].value] = k
                # Connections
                elif o.type.name == "UML - Association":
                    # should already have got attributes relation by names
                    pass
                # other UML objects which may be interesting
                # UML - Note, UML - LargePackage, UML - SmallPackage,
                # UML - Dependency, ...

        edges = {}
        for layer in data.layers:
            for o in layer.objects:
                for c in o.connections:
                    for n in c.connected:
                        if n.type.name not in ("UML - Generalization",
                                               "UML - Realizes"):
                            continue
                        if str(n) in edges:
                            continue
                        edges[str(n)] = None
                        if not (n.handles[0].connected_to and (n.handles[1].
                                                               connected_to)):
                            continue
                        par = n.handles[0].connected_to.object
                        chi = n.handles[1].connected_to.object
                        if not par.type.name == "UML - Class" and\
                           chi.type.name == "UML - Class":
                            continue
                        par_name = par.properties["name"].value
                        chi_name = chi.properties["name"].value
                        if n.type.name == "UML - Generalization":
                            self.klasses[chi_name].AddParrent(par_name)
                        else:
                            self.klasses[chi_name].AddTemplate(par_name)

    def end_render(self):
        # without this we would accumulate info from every pass
        self.attributes = []
        self.operations = []

    def draw_line(self, *args):
        pass

    def draw_string(self, *args):
        pass

    def fill_rect(self, *args):
        pass


class MyPyRenderer(ObjRenderer):
    def __init__(self):
        ObjRenderer.__init__(self)
        self.to_build = {}
        self.built = []

    def header(self):
        s = ""
        s += "import numpy as np\n"
        s += "import datetime\n"
        s += "from spectroscopy.class_factory import _base_class_factory, "
        s += "_buffer_class_factory\n"
        return s

    def build_documentation(self, k):
        """
        Construct sphynx-compatible documentation from datamodel
        comments.
        """
        s = "\t'''\n"
        s += "\t{:s}\n\n".format(k.comment)
        for n, att in k.attributes:
            if n in ['resource_id', 'creation_time', 'modification_time']:
                continue
            if att[0].find('array') != -1:
                dt = ":class:`numpy.ndarray`"
            elif att[0].find('string') != -1:
                dt = "str"
            elif att[0].find('integer') != -1:
                dt = "int"
            else:
                dt = att[0]
            s += "\t:type {0:s}: {1:s}\n".format(n, dt)
            s += "\t:param {0:s}: {1:s}\n".format(n, att[3])
        s += "\t'''"
        return s

    def build_classes(self, k):
        """
        Generate the code that calls the class factory function and
        builds the different variety of classes from the datamodel.
        """
        type_lookup = {'float': 'np.float_', 'integer': 'np.int_',
                       'string': 'np.str_', 'datetime': 'datetime.datetime',
                       'json': 'np.str_', 'set': 'set'}
        d = {}
        s = "\n\n"
        tmp_s = "__{0:s} = _base_class_factory('__{0:s}', '{1:s}',\n"
        s += tmp_s.format(k.name, k.stereotype)
        attributes = []
        for n, att in k.attributes:
            if n in ['resource_id', 'creation_time', 'modification_time']:
                continue
            if att[0].find('reference') != -1:
                continue
            if att[0].find('array') != -1:
                try:
                    dtp = re.match('array of (\w+)s', att[0]).group(1)
                except Exception as e:
                    raise Exception("Can't match {:s}".format(att[0]))
                dt = "(np.ndarray, {:s})".format(type_lookup[dtp])
            else:
                dt = "({:s},)".format(type_lookup[att[0].strip()])
            attributes.append("('{0:s}',{1:s})".format(n, dt))
        i = 0
        cls_props = "\tclass_properties=[\n"
        for a in attributes:
            i += 1
            cls_props += "\t\t{:s}".format(a)
            if i < len(attributes):
                cls_props += ",\n"
        cls_props += "],\n"
        s += cls_props

        references = []
        # Dependencies have to be tracked so the
        # classes are build in the right order
        dependencies = []
        for n, att in k.attributes:
            if att[0].find('reference') != -1:
                if att[0].startswith('array'):
                    dtp = re.match('array of references to (\w+)',
                                   att[0]).group(1)
                    dt = "(np.ndarray, _{:s})".format(dtp)
                    dependencies.append("_{:s}".format(dtp))
                else:
                    try:
                        ref_class = re.match("reference to (\S*)",
                                             att[0]).group(1)
                    except AttributeError:
                        msg = "No reference class found in {:s}".format(att[0])
                        print(msg)
                    dt = "(_{:s},)".format(ref_class)
                    dependencies.append("_{:s}".format(ref_class))
                references.append("('{0:s}',{1:s})".format(n, dt))

        cls_refs = "\tclass_references=[\n"
        if len(references) > 0:
            i = 0
            for r in references:
                i += 1
                cls_refs += "\t\t{:s}".format(r)
                if i < len(references):
                    cls_refs += ",\n"
            cls_refs += "])\n"
        else:
            cls_refs = "\tclass_references=[])\n"
        s += cls_refs

        s += "\n\n"
        s += "__{0:s}Buffer = _buffer_class_factory(\n".format(k.name)
        s += "\t'__{0:s}Buffer', \n".format(k.name)
        s += cls_props
        s += cls_refs
        s += "\n\n"
        s += "class {0:s}Buffer(__{0:s}Buffer):\n".format(k.name)
        s += self.build_documentation(k)
        s += "\n\n"
        s += "class _{0:s}(__{0:s}):\n\t'''\n\t'''\n".format(k.name)
        d['code'] = s
        d['dependencies'] = dependencies
        return d

    def tree_build(self, fh, key):
        """
        Resolve dependencies by following the dependency tree.
        """
        # first check if class has already been built
        if key in self.built:
            return
        # If there aren't any dependencies it's safe to generate
        # the code
        if len(self.to_build[key]['dependencies']) < 1:
            fh.write(self.to_build[key]['code'])
            self.built.append(key)
            return

        for cl in self.to_build[key]['dependencies']:
            if cl not in self.built:
                self.tree_build(fh, cl)
        fh.write(self.to_build[key]['code'])
        self.built.append(key)

    def footer(self):
        s = "\n\n"
        s += "all_classes = ["
        i = 0
        for k in self.built:
            i += 1
            s += "{:s}".format(k)
            if i < len(self.built):
                s += ", "
        s += "]\n"
        return s

    def end_render(self):
        f = open(self.filename, 'w')
        f.write(self.header())
        for sk in self.klasses.keys():
            self.to_build["_{:s}".format(sk)] = \
                    self.build_classes(self.klasses[sk])
        for k in self.to_build.keys():
            self.tree_build(f, k)
        f.write(self.footer())
        f.close()
        self.to_build = {}
        self.built = []
        ObjRenderer.end_render(self)


dia.register_export("Gas chemistry code generation (Python)", "py",
                    MyPyRenderer())
