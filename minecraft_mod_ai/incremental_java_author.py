        while True:
            member_id = member["id"]
            epoch = self._progress_epoch.get(member_id, 0)
            stalled = {op for (owner, op), marker in self._stalled_declarations.items()
                       if owner == member_id and marker == epoch}
            options = operation_options(
                member, scope, body, blocked_declarations | stalled,
                nesting_depth=(len(self.path) + 1) // 2,
            )
            # A small model can otherwise repeatedly invent wrappers, helpers,
            # or evidence reads without writing even one Java action. Restrict
            # the choice schema itself, rather than re-prompting a forbidden
            # answer or silently treating a placeholder as implementation.
            idle = self._nonprogress_choices.get(member_id, 0)
            if idle >= _DIRECT_PROGRESS_AFTER and member["kind"] not in {"class", "record", "enum"}:
                permitted = set(_EXECUTABLE_OPERATIONS)
                if idle < _LOOKUP_EXHAUSTED_AFTER:
                    permitted.add("lookup")
                if body:
                    permitted.add("complete")
                options = [option for option in options if option in permitted]
            if not options:
                raise RuntimeError(
                    f"INCREMENTAL_JAVA_NO_AVAILABLE_OPERATION: member={member_id}"
                )
            self.current_operation = ""
            operation = self.ask(choice("op", options), "kind")["op"]
            self.current_operation = operation
            if operation == "complete":
                break
            if operation == "lookup":
                self.lookup()
                self._record_nonprogress(member)
                continue
            if operation in {*DECLARATION_OPS, "constructor", "override"}:
                author_declaration(self, operation)
                # Do not recursively enter a newly declared helper. The work
                # queue is drained only after the current member completes.
                if operation in {"method", "type", "constructor", "override"} and member["kind"] not in {"class", "record", "enum"}:
                    self._stalled_declarations[(member_id, operation)] = epoch
                if self.last_declaration.get("outcome") in {"reused", "active"}:
                    blocked_declarations.add(operation)
                    # The same active helper may not be requested again merely
                    # by opening a fresh nested block. Reset only after actual
                    # executable work, not another declaration or condition.
                    if operation == "method" and self.last_declaration["outcome"] == "active":
                        self._stalled_declarations[(member_id, operation)] = epoch
                self._record_nonprogress(member)
                continue
            progress_before = self._progress_epoch.get(member_id, 0)
            node = author_operation(self, operation)
            if node is None:
                self._record_nonprogress(member)
                continue
            body.append(node)
            self.history[node["id"]] = node
            self.rejection = {}
            blocked_declarations.clear()
            if operation in _EXECUTABLE_OPERATIONS:
                self._record_progress(member)
            elif self._progress_epoch.get(member_id, 0) == progress_before:
                self._record_nonprogress(member)
            if operation in {"return", "throw", "break", "continue"}:
                break
        self.member, self.scope, self.path, self.block, self.current_operation = previous
        return body

    def finish(self):
        from .incremental_java_progress import finish_member
        # FIFO iteration keeps helper -> helper chains off the Python stack.
        # Every declared method and listener is authored before final rendering.
        while self.pending:
            member = self.pending.popleft()
            if member.get("implementation_status") == "complete":
                continue
            finish_member(self, member)
        unfinished = [m["id"] for m in self.program["members"]
                      if m.get("implementation_status") in {"pending", "authoring"}]
        if unfinished:
            raise RuntimeError(f"INCREMENTAL_JAVA_UNFINISHED_MEMBERS: {unfinished}")
        _assert_live_behavior_entrypoints(self.program)
        return self.render()

    def render(self):
        from .host_java_imports import host_add_verified_imports
        source, mapping = render_program(self.program)
        final = host_add_verified_imports(source, symbols=self.api.symbols,
            owner=self.program["package"] + "." + self.program["class_name"], target_path=self.spec["target_path"])
        offset = len(final.splitlines()) - len(source.splitlines())
        self.line_map = {line + (offset if line > 1 else 0): value for line, value in mapping.items()}
        return final, self.program


def compose_incremental_feature(router, spec, messages, *, construction_context, behaviors):
    if not behaviors:
        raise ValueError("JAVA_INCREMENTAL_BEHAVIORS_REQUIRED")
    return IncrementalJavaBuilder(router, spec, messages, construction_context, behaviors).finish()
        try:
            while True:
                member_id = member["id"]
                epoch = self._progress_epoch.get(member_id, 0)
                stalled = {op for (owner, op), marker in self._stalled_declarations.items()
                           if owner == member_id and marker == epoch}
                options = operation_options(
                    member, scope, body, blocked_declarations | stalled,
                    nesting_depth=(len(self.path) + 1) // 2,
                )
                # A small model can otherwise repeatedly invent wrappers, helpers,
                # or evidence reads without writing even one Java action. Restrict
                # the choice schema itself, rather than re-prompting a forbidden
                # answer or silently treating a placeholder as implementation.
                idle = self._nonprogress_choices.get(member_id, 0)
                if idle >= _DIRECT_PROGRESS_AFTER and member["kind"] not in {"class", "record", "enum"}:
                    permitted = set(_EXECUTABLE_OPERATIONS)
                    if idle < _LOOKUP_EXHAUSTED_AFTER:
                        permitted.add("lookup")
                    if body:
                        permitted.add("complete")
                    options = [option for option in options if option in permitted]
                if not options:
                    raise RuntimeError(
                        f"INCREMENTAL_JAVA_NO_AVAILABLE_OPERATION: member={member_id}"
                    )
                self.current_operation = ""
                operation = self.ask(choice("op", options), "kind")["op"]
                self.current_operation = operation
                if operation == "complete":
                    break
                if operation == "lookup":
                    self.lookup()
                    self._record_nonprogress(member)
                    continue
                if operation in {*DECLARATION_OPS, "constructor", "override"}:
                    author_declaration(self, operation)
                    # Do not recursively enter a newly declared helper. The work
                    # queue is drained only after the current member completes.
                    if operation in {"method", "type", "constructor", "override"} and member["kind"] not in {"class", "record", "enum"}:
                        self._stalled_declarations[(member_id, operation)] = epoch
                    if self.last_declaration.get("outcome") in {"reused", "active"}:
                        blocked_declarations.add(operation)
                        # The same active helper may not be requested again merely
                        # by opening a fresh nested block. Reset only after actual
                        # executable work, not another declaration or condition.
                        if operation == "method" and self.last_declaration["outcome"] == "active":
                            self._stalled_declarations[(member_id, operation)] = epoch
                    self._record_nonprogress(member)
                    continue
                progress_before = self._progress_epoch.get(member_id, 0)
                node = author_operation(self, operation)
                if node is None:
                    self._record_nonprogress(member)
                    continue
                body.append(node)
                self.history[node["id"]] = node
                self.rejection = {}
                blocked_declarations.clear()
                if operation in _EXECUTABLE_OPERATIONS:
                    self._record_progress(member)
                elif self._progress_epoch.get(member_id, 0) == progress_before:
                    self._record_nonprogress(member)
                if operation in {"return", "throw", "break", "continue"}:
                    break
        finally:
            self.member, self.scope, self.path, self.block, self.current_operation = previous
        return body

    def finish(self):
        from .incremental_java_progress import finish_member
        # FIFO iteration keeps helper -> helper chains off the Python stack.
        # Every declared method and listener is authored before final rendering.
        while self.pending:
            member = self.pending.popleft()
            if member.get("implementation_status") == "complete":
                continue
            finish_member(self, member)
        unfinished = [m["id"] for m in self.program["members"]
                      if m.get("implementation_status") in {"pending", "authoring"}]
        if unfinished:
            raise RuntimeError(f"INCREMENTAL_JAVA_UNFINISHED_MEMBERS: {unfinished}")
        _assert_live_behavior_entrypoints(self.program)
        return self.render()

    def render(self):
        from .host_java_imports import host_add_verified_imports
        source, mapping = render_program(self.program)
        final = host_add_verified_imports(source, symbols=self.api.symbols,
            owner=self.program["package"] + "." + self.program["class_name"], target_path=self.spec["target_path"])
        offset = len(final.splitlines()) - len(source.splitlines())
        self.line_map = {line + (offset if line > 1 else 0): value for line, value in mapping.items()}
        return final, self.program


def compose_incremental_feature(router, spec, messages, *, construction_context, behaviors):
    if not behaviors:
        raise ValueError("JAVA_INCREMENTAL_BEHAVIORS_REQUIRED")
    return IncrementalJavaBuilder(router, spec, messages, construction_context, behaviors).finish()
