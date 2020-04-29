patch_string = '''{
	"boxes" : [ 		{
			"box" : 			{
				"maxclass" : "ezdac~",
				"patching_rect" : [ 142.0, 256.0, 45.0, 45.0 ],
				"numinlets" : 2,
				"id" : "obj-3",
				"numoutlets" : 0,
				"style" : ""
			}

		}
, 		{
			"box" : 			{
				"maxclass" : "newobj",
				"text" : "*~ 0.5",
				"outlettype" : [ "signal" ],
				"patching_rect" : [ 142.0, 190.0, 42.0, 22.0 ],
				"numinlets" : 2,
				"id" : "obj-2",
				"numoutlets" : 1,
				"style" : ""
			}

		}
, 		{
			"box" : 			{
				"maxclass" : "newobj",
				"text" : "cycle~ 440",
				"outlettype" : [ "signal" ],
				"patching_rect" : [ 142.0, 139.0, 68.0, 22.0 ],
				"numinlets" : 2,
				"id" : "obj-1",
				"numoutlets" : 1,
				"style" : ""
			}

		}
 ],
	"lines" : [ 		{
			"patchline" : 			{
				"source" : [ "obj-2", 0 ],
				"destination" : [ "obj-3", 0 ]
			}

		}
, 		{
			"patchline" : 			{
				"source" : [ "obj-1", 0 ],
				"destination" : [ "obj-2", 0 ]
			}

		}
 ],
	"appversion" : 	{
		"major" : 7,
		"minor" : 3,
		"revision" : 6,
		"architecture" : "x64",
		"modernui" : 1
	}

}
'''

import json


class Patch:
    def __init__(self, patch_string):
        self.patch_dict = json.loads(patch_string)
        ezdacs = self.objects_by_class('ezdac~')
        assert len(ezdacs) == 1
        print(self.parents(ezdacs[0], inlet=0))

    def objects_by_class(self, object_class):
        result = []
        for object in self.patch_dict['boxes']:
            if object['box']['maxclass'] == object_class:
                result.append(object)
        return result

    def parents(self, object, inlet):
        id = object['box']['id']
        result = []
        for patchchord in self.patch_dict['lines']:
            if patchchord['patchline']['destination'][0] == id and patchchord['patchline']['destination'][1] == inlet:
                result.append(self.object_by_id(patchchord['patchline']['source'][0]))
        return result

    def object_by_id(self, id):
        for object in self.patch_dict['boxes']:
            if object['box']['id'] == id:
                return object
        return None


Patch(patch_string)
