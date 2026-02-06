import argparse

from llm_config.config.api import all_tables, initialize, ATTRS_MAP


def add_parse(subparsers):
    parser  = subparsers.add_parser('add', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--from-file', type=str, help='')
    parser.add_argument('--from-base', type=str, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--attributes', type=str, help='')
    parser.add_argument('--base-layer', type=str, help='')


def update_parse(subparsers):
    parser  = subparsers.add_parser('update', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--from-file', type=str, help='')
    parser.add_argument('--from-base', type=str, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--attributes', type=str, help='')


def delete_parse(subparsers):
    parser  = subparsers.add_parser('delete', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--all', action='store_true', help='')


def show_parse(subparsers):
    parser = subparsers.add_parser('show', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--detailed', action='store_true', help='')
    parser.add_argument('--attribute', type=str, help='')
    parser.add_argument('--list-attributes', action='store_true', help='')
    parser.add_argument('--list', action='store_true', help='')


def parse_args():
    parser = argparse.ArgumentParser(description='HLO Config')

    subparsers = parser.add_subparsers(dest='command', help='')
    add_parse(subparsers)
    update_parse(subparsers)
    delete_parse(subparsers)
    show_parse(subparsers)

    args = parser.parse_args()
    return args


def execute_add(args):
    from llm_config.config.api import config_from_file, config_from_base, add_all, query, CONFIG_MAP
    from llm_config.config.base import create_session
    
    if not args.module:
        raise RuntimeError("--module parameter is required for add command")
    
    if args.from_file and args.from_base:
        raise RuntimeError("--from-file and --from-base cannot be used together")
    
    if args.from_file:
        configs = config_from_file(args.from_file, args.module)
        # Check if any config with the same name already exists
        config_type = CONFIG_MAP.get(args.module)
        for cfg in configs:
            existing = query(cfg.name, config_type)
            if existing is not None:
                raise ValueError(f"Config with name '{cfg.name}' already exists")
        add_all(configs) # add_all uses session.add_all, which is fine for transient objects
        print(f"Successfully added {len(configs)} config(s) from file {args.from_file}")
    elif args.from_base:
        if not args.attributes:
            raise RuntimeError("--attributes parameter is required when using --from-base")
        
        # config_from_base returns a list of objects:
        # - The new ModelConfig (transient)
        # - Any new component configs (transient)
        # - Any existing component configs that were updated (detached persistent)
        configs_to_process = config_from_base(args.module, args.from_base, args.attributes, args.base_layer)
        
        config_type = CONFIG_MAP.get(args.module)
        # Find the main new config (e.g., ModelConfig) to get its name for the success message
        main_config = next((c for c in configs_to_process if isinstance(c, config_type)), None)
        if main_config is None:
            raise RuntimeError("Could not find the main config object in the list returned by config_from_base.")
        new_config_name = main_config.name
        
        with create_session() as session:
            for cfg_item in configs_to_process:
                # session.merge handles both transient (new) and detached persistent (existing, updated) objects.
                # It will either add a new object, or re-associate an existing one and copy its state.
                session.merge(cfg_item)
            session.commit()
        
        print(f"Successfully created config '{new_config_name}' from base '{args.from_base}'")
    else:
        raise RuntimeError("Either --from-file or --from-base must be specified")


def execute_update(args):
    from llm_config.config.api import update, query, CONFIG_MAP, _parse_update_string, _apply_updates, _check_dependencies_and_confirm
    
    if not args.name:
        raise RuntimeError("--name parameter is required for update command")
    
    if not args.attributes:
        raise RuntimeError("--attributes parameter is required for update command")
    
    config_type = CONFIG_MAP.get(args.module)
    assert config_type is not None, f"`{args.module}` is not a valid module name"
    
    # Enable eager loading for model to avoid DetachedInstanceError on nested updates
    eager_load = (args.module == 'model')
    
    # Check if config exists
    obj = query(args.name, config_type, eager_load=eager_load)
    if obj is None:
        raise ValueError(f"Config with name '{args.name}' not found")
    
    # Parse updates
    updates = _parse_update_string(args.attributes)
    
    # Check if trying to update name (not recommended)
    if 'name' in updates:
        print("Warning: Updating 'name' is not recommended. Consider creating a new config instead.")
        # Use _check_dependencies_and_confirm for consistency, though it's not a shared component issue here
        if not _check_dependencies_and_confirm(args.name, config_type, "Updating"):
            return
    
    # Check dependencies if updating attributes of a model's component
    if args.module == 'model':
        for component_name in ATTRS_MAP.keys():
            # Check if any attribute of this component is being updated, but not the name reference itself
            is_attribute_update = any(k.startswith(f"{component_name}.") and k != f"{component_name}.name" for k in updates)

            if is_attribute_update:
                component_attr = ATTRS_MAP.get(component_name)
                component_obj = getattr(obj, component_attr, None)
                
                if component_obj:
                    # Pass the current model's name to exclude it from the dependency check
                    if not _check_dependencies_and_confirm(component_obj.name, CONFIG_MAP[component_name], "Updating", exclude_model_name=args.name):
                        print("Update cancelled by user.")
                        return
    
    # Apply updates
    _apply_updates(obj, updates, config_type)
    
    # Save to database
    from llm_config.config.base import create_session
    with create_session() as session:
        session.add(obj) # obj is already loaded and attached (or merged by query if eager_load was true)
        session.commit()
    
    print(f"Successfully updated config '{args.name}'")


def execute_delete(args):
    from llm_config.config.api import delete, query, CONFIG_MAP, _check_dependencies_and_confirm
    from llm_config.config.base import create_session
    
    if not args.name:
        raise RuntimeError("--name parameter is required for delete command")
    
    config_type = CONFIG_MAP.get(args.module)
    assert config_type is not None, f"`{args.module}` is not a valid module name"
    
    # Check if config exists
    obj = query(args.name, config_type)
    if obj is None:
        raise ValueError(f"Config with name '{args.name}' not found")
    
    # Check dependencies for top-level config deletion
    if not _check_dependencies_and_confirm(args.name, config_type, "Deleting"):
        print("Delete cancelled by user")
        return
    
    # Handle model deletion with --all flag
    if args.module == 'model' and args.all:
        # Delete all associated components
        with create_session() as session:
            session.add(obj) # Re-add to session for deletion
            # Get all component references
            components_to_delete = []
            for component_name, attr_name in [('mla', 'mla_config'), ('mha', 'mha_config'), 
                                             ('mlp', 'mlp_config'), ('moe', 'moe_config'),
                                             ('rope', 'rope_config'), ('norm', 'rmsnorm_config')]:
                component_obj = getattr(obj, attr_name, None)
                if component_obj:
                    # Check if component is used by other models
                    if _check_dependencies_and_confirm(component_obj.name, CONFIG_MAP[component_name], "Deleting", exclude_model_name=args.name):
                        components_to_delete.append(component_obj)
                    else:
                        print(f"Deletion of component '{component_obj.name}' cancelled by user. Model '{args.name}' will still be deleted.")
            
            # Delete components
            for component in components_to_delete:
                session.delete(component)
            
            # Delete model
            session.delete(obj)
            session.commit()
        print(f"Successfully deleted model '{args.name}' and associated components")
    else:
        # Regular delete
        delete(args.name, config_type)
        print(f"Successfully deleted config '{args.name}'")


def execute_show(args):
    from llm_config.config.api import show_list, show, show_attributes, CONFIG_MAP
    
    if not args.module:
        raise RuntimeError("--module parameter is required for show command")
    
    config_type = CONFIG_MAP.get(args.module)
    assert config_type is not None, f"`{args.module}` is not a valid module name"
    
    if args.list_attributes:
        show_attributes(args.module)
    elif args.list:
        show_list(args.module)
    elif args.name:
        # show function now handles nested attributes internally
        show(args.name, args.module, attr_name=args.attribute, is_detail=args.detailed)
    else:
        raise RuntimeError("Either --list, --list-attributes, or --name must be specified")


def main():
    initialize()

    args = parse_args()
    execute_func = globals().get(f"execute_{args.command}")
    assert execute_func is not None and callable(execute_func)
    execute_func(args)


if __name__ == '__main__':
    main()
