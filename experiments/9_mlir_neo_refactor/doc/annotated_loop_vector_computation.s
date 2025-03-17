	.text
	.file	"loop_vector_computation.llvm"


	.globl	vector_calc                     # -- Begin function vector_calc
	.p2align	4, 0x90
	.type	vector_calc,@function
vector_calc:                            # @vector_calc
	.cfi_startproc
# %bb.0:                                # %entry

	movl	$0, -4(%rsp)       #  %i = Stack[-4] = $0
	.p2align	4, 0x90
.LBB0_1:                                # %loop_cond
                                        # =>This Inner Loop Header: Depth=1
	movl	-4(%rsp), %eax
	cmpl	%ecx, %eax
	jge	.LBB0_3
# %bb.2:                                # %loop_body
                                        #   in Loop: Header=BB0_1 Depth=1
	leal	1(%rax), %r8d
	movslq	%eax, %r9
	decl	%eax
	movslq	%eax, %r10
	movslq	%r8d, %r8
	movl	(%rdi,%r9,4), %eax
	addl	(%rsi,%r10,4), %eax
	subl	(%rdi,%r8,4), %eax
	movl	%eax, (%rdx,%r9,4)
	movl	%r8d, -4(%rsp)
	jmp	.LBB0_1
.LBB0_3:                                # %loop_end
	retq
.Lfunc_end0:
	.size	vector_calc, .Lfunc_end0-vector_calc
	.cfi_endproc
                                        # -- End function




	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movl	$100, 20(%rsp)
	leaq	16(%rsp), %rdi
	leaq	8(%rsp), %rsi
	movq	%rsp, %rdx
	movl	$100, %ecx
	callq	vector_calc@PLT
	xorl	%eax, %eax
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
