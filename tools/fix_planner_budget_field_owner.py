from pathlib import Path

path = Path('minecraft_mod_ai/pre_design_research_pipeline.py')
text = path.read_text(encoding='utf-8')
old = '''    budget = request_message_budget(config, ())\n    for section_id, fields, _properties in agentic._SECTION_SPECS:\n        messages = agentic._section_messages(\n            prompt=prompt,\n            section_id=section_id,\n            fields=fields,\n            research=research,\n        )\n        prepared = _inject_system_context(\n            messages, _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT\n        )\n        if _canonical_size(prepared) > budget:\n            return False\n    return True\n'''
new = '''    budget = request_message_budget(config, ())\n    for section_id, fields, _properties in agentic._SECTION_SPECS:\n        for field in fields:\n            messages = agentic._field_messages(\n                prompt=prompt,\n                section_id=section_id,\n                field=field,\n                research=research,\n            )\n            prepared = _inject_system_context(\n                messages, _REPOSITORY_MAIN_ONLY_SYSTEM_CONTEXT\n            )\n            if _canonical_size(prepared) > budget:\n                return False\n    return True\n'''
if old not in text:
    raise RuntimeError('old section-owned context budget block not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
