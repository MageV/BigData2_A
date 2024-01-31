import contextvars

file_counter_ctx: contextvars.ContextVar[int] = contextvars.ContextVar("FileCounter", default=0)
